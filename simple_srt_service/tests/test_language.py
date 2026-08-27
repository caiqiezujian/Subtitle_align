from simple_srt_service.language import detect_language


def test_detect_language():
    assert detect_language("这是中文字幕") == "Chinese"
    assert detect_language("これは日本語です") == "Japanese"
    assert detect_language("This is an English subtitle.") == "English"

