const $ = (selector) => document.querySelector(selector);

const form = $("#alignForm");
const submitButton = $("#submitButton");
const resultPanel = $("#resultPanel");
const progressBar = $("#progressBar");
const progressText = $("#progressText");
const stageText = $("#stageText");
const statusPill = $("#statusPill");
const errorMessage = $("#errorMessage");
const resultSummary = $("#resultSummary");
const downloadRow = $("#downloadRow");
const reviewStudio = $("#reviewStudio");
const videoPlayer = $("#videoPlayer");
const audioPlayer = $("#audioPlayer");
const audioStage = $("#audioStage");
const captionOverlay = $("#captionOverlay");
const cueList = $("#cueList");
let activeJobId = null;
let pollTimer = null;
let activeMedia = null;
let mediaObjectUrl = null;
let subtitleCues = [];
let activeCueIndex = -1;
let reviewJobId = null;
let completedJob = null;

function parseSrtTimestamp(value) {
  const match = String(value).trim().match(/(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})/);
  if (!match) return null;
  return Number(match[1]) * 3600 + Number(match[2]) * 60 + Number(match[3]) + Number(match[4]) / 1000;
}

function parseSrt(source) {
  return String(source)
    .replace(/^\uFEFF/, "")
    .replace(/\r/g, "")
    .trim()
    .split(/\n{2,}/)
    .map((block, fallbackIndex) => {
      const lines = block.split("\n");
      const timingIndex = lines.findIndex((line) => line.includes("-->"));
      if (timingIndex < 0) return null;
      const [rawStart, rawEnd] = lines[timingIndex].split("-->");
      const start = parseSrtTimestamp(rawStart);
      const end = parseSrtTimestamp(rawEnd);
      const text = lines.slice(timingIndex + 1).join("\n").trim();
      if (start === null || end === null || end < start || !text) return null;
      const sequence = Number.parseInt(lines[timingIndex - 1], 10);
      return { index: Number.isFinite(sequence) ? sequence : fallbackIndex + 1, start, end, text };
    })
    .filter(Boolean);
}

function formatClock(seconds, includeMillis = false) {
  const safe = Math.max(0, Number(seconds) || 0);
  const totalMillis = Math.round(safe * 1000);
  const hours = Math.floor(totalMillis / 3_600_000);
  const minutes = Math.floor((totalMillis % 3_600_000) / 60_000);
  const secs = Math.floor((totalMillis % 60_000) / 1000);
  const base = [hours, minutes, secs].map((part) => String(part).padStart(2, "0")).join(":");
  return includeMillis ? `${base}.${String(totalMillis % 1000).padStart(3, "0")}` : base;
}

function findActiveCue(time) {
  let low = 0;
  let high = subtitleCues.length - 1;
  let candidate = -1;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    if (subtitleCues[middle].start <= time) {
      candidate = middle;
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }
  return candidate >= 0 && time < subtitleCues[candidate].end ? candidate : -1;
}

function setActiveCue(index) {
  if (index === activeCueIndex) return;
  const previous = cueList.querySelector(".cue-item.active");
  if (previous) previous.classList.remove("active");
  activeCueIndex = index;
  const cue = subtitleCues[index];
  captionOverlay.textContent = cue?.text || "";
  captionOverlay.classList.toggle("visible", Boolean(cue) && $("#captionToggle").checked);
  $("#cueCounter").textContent = cue ? `${index + 1} / ${subtitleCues.length}` : `— / ${subtitleCues.length}`;
  if (!cue) return;
  const current = cueList.querySelector(`[data-cue-index="${index}"]`);
  if (current) {
    current.classList.add("active");
    current.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
}

function syncPlayer() {
  if (!activeMedia) return;
  $("#currentTime").textContent = formatClock(activeMedia.currentTime);
  setActiveCue(findActiveCue(activeMedia.currentTime));
}

function renderCueList() {
  cueList.replaceChildren();
  const fragment = document.createDocumentFragment();
  subtitleCues.forEach((cue, index) => {
    const button = document.createElement("button");
    button.className = "cue-item";
    button.type = "button";
    button.dataset.cueIndex = String(index);
    button.setAttribute("aria-label", `跳转到 ${formatClock(cue.start, true)}：${cue.text}`);

    const marker = document.createElement("span");
    marker.className = "cue-marker";
    marker.textContent = String(cue.index).padStart(2, "0");
    const body = document.createElement("span");
    body.className = "cue-body";
    const time = document.createElement("small");
    time.textContent = `${formatClock(cue.start, true)} — ${formatClock(cue.end, true)}`;
    const text = document.createElement("strong");
    text.textContent = cue.text;
    body.append(time, text);
    button.append(marker, body);
    button.addEventListener("click", () => {
      if (!activeMedia) return;
      const shouldResume = !activeMedia.paused && !activeMedia.ended;
      activeMedia.currentTime = cue.start;
      setActiveCue(index);
      if (shouldResume) activeMedia.play().catch(() => {});
    });
    fragment.append(button);
  });
  cueList.append(fragment);
  $("#cueCounter").textContent = `— / ${subtitleCues.length}`;
}

function selectMediaPlayer(file) {
  if (mediaObjectUrl) URL.revokeObjectURL(mediaObjectUrl);
  mediaObjectUrl = URL.createObjectURL(file);
  videoPlayer.pause();
  audioPlayer.pause();
  videoPlayer.removeAttribute("src");
  audioPlayer.removeAttribute("src");

  const isVideo = file.type.startsWith("video/") || /\.(mp4|mkv|webm|mov|avi|wmv|m4v)$/i.test(file.name);
  activeMedia = isVideo ? videoPlayer : audioPlayer;
  videoPlayer.classList.toggle("hidden", !isVideo);
  audioStage.classList.toggle("hidden", isVideo);
  $("#playerEmpty").classList.add("hidden");
  $("#reviewMediaName").textContent = file.name;
  activeMedia.src = mediaObjectUrl;
  activeMedia.playbackRate = Number($("#playbackRate").value);
  activeMedia.load();
}

async function initializeReview(job) {
  if (reviewJobId === job.id) return;
  const mediaFile = $("#mediaInput").files[0];
  if (!mediaFile) throw new Error("浏览器中已找不到原始媒体，请重新创建任务后验收。");
  reviewJobId = job.id;
  reviewStudio.classList.remove("hidden");
  $("#reviewReadyText").textContent = "正在读取最终 SRT";
  $("#playerNote").classList.remove("error");
  selectMediaPlayer(mediaFile);

  const response = await fetch(`/api/jobs/${job.id}/download/srt`, { headers: apiHeaders() });
  if (!response.ok) throw new Error((await response.json()).detail || "读取 SRT 失败");
  subtitleCues = parseSrt(await response.text());
  if (!subtitleCues.length) throw new Error("SRT 中没有可播放的有效字幕，请检查对齐结果。");
  activeCueIndex = -1;
  renderCueList();
  $("#reviewReadyText").textContent = `${subtitleCues.length} 条字幕已就绪`;
  $("#playerNote").textContent = "点击右侧任意字幕可跳转到对应时间，播放时当前字幕会自动高亮。";
}

function showReviewError(error) {
  reviewJobId = null;
  reviewStudio.classList.remove("hidden");
  $("#reviewReadyText").textContent = "验收区准备失败";
  $("#playerNote").textContent = error.message || "无法载入播放器，请重新创建任务。";
  $("#playerNote").classList.add("error");
}

function apiHeaders() {
  const key = $("#apiKey").value.trim();
  if (key) sessionStorage.setItem("subalign_api_key", key);
  return key ? { "X-API-Key": key } : {};
}

function setupDropzone(zoneSelector, inputSelector, nameSelector) {
  const zone = $(zoneSelector);
  const input = $(inputSelector);
  const name = $(nameSelector);
  const showFile = () => {
    if (!input.files[0]) return;
    name.textContent = input.files[0].name;
    zone.classList.add("selected");
  };
  input.addEventListener("change", showFile);
  ["dragenter", "dragover"].forEach((event) => zone.addEventListener(event, () => zone.classList.add("dragover")));
  ["dragleave", "drop"].forEach((event) => zone.addEventListener(event, () => zone.classList.remove("dragover")));
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    if (!event.dataTransfer.files.length) return;
    const transfer = new DataTransfer();
    transfer.items.add(event.dataTransfer.files[0]);
    input.files = transfer.files;
    showFile();
  });
  zone.addEventListener("dragover", (event) => event.preventDefault());
}

setupDropzone("#mediaZone", "#mediaInput", "#mediaName");
setupDropzone("#transcriptZone", "#transcriptInput", "#transcriptName");
$("#apiKey").value = sessionStorage.getItem("subalign_api_key") || "";

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const health = await response.json();
    const dot = $("#healthDot");
    if (health.status === "ok") {
      dot.className = "ok";
      $("#healthText").textContent = health.v4_flash ? "服务与 v4-flash 已就绪" : "对齐服务已就绪";
    } else {
      dot.className = "warn";
      $("#healthText").textContent = "服务配置待检查";
    }
    if (!health.v4_flash) {
      $("#useFlash").disabled = true;
      $("#useFlash").closest("label").title = "服务器尚未配置 v4-flash";
    }
  } catch (_) {
    $("#healthDot").className = "warn";
    $("#healthText").textContent = "暂时无法连接服务";
  }
}

function renderJob(job) {
  const progress = Number(job.progress || 0);
  progressBar.style.width = `${progress}%`;
  progressText.textContent = `${progress}%`;
  stageText.textContent = job.stage || "处理中";
  const labels = { queued: "排队中", running: "处理中", completed: "已完成", failed: "失败" };
  statusPill.textContent = labels[job.status] || job.status;
  const phases = [...document.querySelectorAll("#progressPhases span")];
  phases.forEach((phase, index) => {
    const start = Number(phase.dataset.start);
    const nextStart = index + 1 < phases.length ? Number(phases[index + 1].dataset.start) : 101;
    phase.classList.toggle("complete", progress >= nextStart || job.status === "completed");
    phase.classList.toggle("active", progress >= start && progress < nextStart && job.status !== "completed");
  });

  if (job.detected_format || job.line_count) {
    resultSummary.classList.remove("hidden");
    $("#detectedFormat").textContent = (job.detected_format || "—").toUpperCase();
    $("#lineCount").textContent = job.line_count ?? "—";
    $("#alignedCount").textContent = job.aligned_count ?? "—";
  }
  if (job.status === "completed") {
    completedJob = job;
    downloadRow.classList.remove("hidden");
    submitButton.disabled = false;
    submitButton.querySelector("span").textContent = "开始对齐";
    clearTimeout(pollTimer);
    initializeReview(job).catch(showReviewError);
  } else if (job.status === "failed") {
    errorMessage.textContent = job.error || "处理失败，请检查文件后重试。";
    errorMessage.classList.remove("hidden");
    submitButton.disabled = false;
    submitButton.querySelector("span").textContent = "重新提交";
    clearTimeout(pollTimer);
  }
}

async function pollJob() {
  if (!activeJobId) return;
  try {
    const response = await fetch(`/api/jobs/${activeJobId}`, { headers: apiHeaders() });
    if (!response.ok) throw new Error((await response.json()).detail || "读取进度失败");
    const job = await response.json();
    renderJob(job);
    if (!["completed", "failed"].includes(job.status)) pollTimer = setTimeout(pollJob, 1800);
  } catch (error) {
    stageText.textContent = error.message;
    pollTimer = setTimeout(pollJob, 3500);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!$("#mediaInput").files[0] || !$("#transcriptInput").files[0]) return;
  submitButton.disabled = true;
  submitButton.querySelector("span").textContent = "正在上传";
  resultPanel.classList.remove("hidden");
  resultSummary.classList.add("hidden");
  downloadRow.classList.add("hidden");
  errorMessage.classList.add("hidden");
  progressBar.style.width = "2%";
  progressText.textContent = "2%";
  stageText.textContent = "正在安全上传文件";
  statusPill.textContent = "上传中";
  resultPanel.scrollIntoView({ behavior: "smooth", block: "center" });

  const data = new FormData();
  data.append("media", $("#mediaInput").files[0]);
  data.append("transcript", $("#transcriptInput").files[0]);
  data.append("language", $("#language").value);
  data.append("text_field", $("#textField").value);
  data.append("use_flash", $("#useFlash").checked);
  data.append("asr_context", $("#asrContext").value);
  data.append("local_refine", $("#localRefine").checked);

  try {
    const response = await fetch("/api/jobs", { method: "POST", headers: apiHeaders(), body: data });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "任务创建失败");
    activeJobId = payload.id;
    renderJob(payload);
    if (!["completed", "failed"].includes(payload.status)) pollJob();
  } catch (error) {
    renderJob({ status: "failed", progress: 0, stage: "提交失败", error: error.message });
  }
});

document.querySelectorAll(".download-button").forEach((button) => {
  button.addEventListener("click", async () => {
    if (!activeJobId) return;
    try {
      const response = await fetch(`/api/jobs/${activeJobId}/download/${button.dataset.kind}`, { headers: apiHeaders() });
      if (!response.ok) throw new Error((await response.json()).detail || "下载失败");
      const blob = await response.blob();
      const disposition = response.headers.get("content-disposition") || "";
      const encoded = disposition.match(/filename\*=utf-8''([^;]+)/i)?.[1];
      const fallback = `aligned.${button.dataset.kind}`;
      const filename = encoded ? decodeURIComponent(encoded) : fallback;
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      errorMessage.textContent = error.message;
      errorMessage.classList.remove("hidden");
    }
  });
});

[videoPlayer, audioPlayer].forEach((media) => {
  media.addEventListener("timeupdate", syncPlayer);
  media.addEventListener("seeked", syncPlayer);
  media.addEventListener("loadedmetadata", () => {
    if (media !== activeMedia) return;
    $("#totalTime").textContent = formatClock(media.duration);
  });
  media.addEventListener("durationchange", () => {
    if (media === activeMedia && Number.isFinite(media.duration)) {
      $("#totalTime").textContent = formatClock(media.duration);
    }
  });
  media.addEventListener("error", () => {
    if (media !== activeMedia) return;
    $("#playerNote").textContent = "当前浏览器不能直接播放这种封装或编码，请使用 MP4、WebM、MP3、WAV 等常见格式进行页面验收。";
    $("#playerNote").classList.add("error");
  });
  media.addEventListener("play", () => {
    if (media === audioPlayer) audioStage.classList.add("playing");
  });
  media.addEventListener("pause", () => {
    if (media === audioPlayer) audioStage.classList.remove("playing");
  });
});

$("#captionToggle").addEventListener("change", () => {
  captionOverlay.classList.toggle("visible", $("#captionToggle").checked && activeCueIndex >= 0);
});

$("#playbackRate").addEventListener("change", () => {
  if (activeMedia) activeMedia.playbackRate = Number($("#playbackRate").value);
});

$("#fullscreenButton").addEventListener("click", async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await $("#playerFrame").requestFullscreen();
  } catch (_) {
    $("#playerNote").textContent = "浏览器未允许全屏，可继续在当前页面验收。";
  }
});

$("#openReview").addEventListener("click", () => {
  if (completedJob && reviewJobId !== completedJob.id) {
    initializeReview(completedJob).catch(showReviewError);
  }
  reviewStudio.scrollIntoView({ behavior: "smooth", block: "start" });
  activeMedia?.focus();
});

$("#newTask").addEventListener("click", () => {
  clearTimeout(pollTimer);
  activeJobId = null;
  reviewJobId = null;
  completedJob = null;
  subtitleCues = [];
  activeCueIndex = -1;
  if (activeMedia) activeMedia.pause();
  if (mediaObjectUrl) URL.revokeObjectURL(mediaObjectUrl);
  mediaObjectUrl = null;
  activeMedia = null;
  videoPlayer.removeAttribute("src");
  audioPlayer.removeAttribute("src");
  audioStage.classList.remove("playing");
  form.reset();
  $("#apiKey").value = sessionStorage.getItem("subalign_api_key") || "";
  $("#localRefine").checked = true;
  ["#mediaZone", "#transcriptZone"].forEach((selector) => $(selector).classList.remove("selected"));
  $("#mediaName").textContent = "MP3 · WAV · MP4 · MKV · WEBM";
  $("#transcriptName").textContent = "TXT · SRT · JSONL · JSON · CSV · TSV";
  resultPanel.classList.add("hidden");
  reviewStudio.classList.add("hidden");
  captionOverlay.classList.remove("visible");
  captionOverlay.textContent = "";
  cueList.replaceChildren();
  window.scrollTo({ top: form.getBoundingClientRect().top + window.scrollY - 90, behavior: "smooth" });
});

checkHealth();
