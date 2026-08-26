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
let activeJobId = null;
let pollTimer = null;

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
    downloadRow.classList.remove("hidden");
    submitButton.disabled = false;
    submitButton.querySelector("span").textContent = "开始对齐";
    clearTimeout(pollTimer);
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
    pollJob();
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

$("#newTask").addEventListener("click", () => {
  clearTimeout(pollTimer);
  activeJobId = null;
  form.reset();
  $("#apiKey").value = sessionStorage.getItem("subalign_api_key") || "";
  $("#localRefine").checked = true;
  ["#mediaZone", "#transcriptZone"].forEach((selector) => $(selector).classList.remove("selected"));
  $("#mediaName").textContent = "MP3 · WAV · MP4 · MKV · WEBM";
  $("#transcriptName").textContent = "TXT · SRT · JSONL · JSON · CSV · TSV";
  resultPanel.classList.add("hidden");
  window.scrollTo({ top: form.getBoundingClientRect().top + window.scrollY - 90, behavior: "smooth" });
});

checkHealth();
