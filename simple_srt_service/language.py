from __future__ import annotations


def detect_language(text: str) -> str:
    """Choose one of the three languages accepted by the alignment core."""
    for char in text:
        code = ord(char)
        if 0x3040 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF:
            return "Japanese"
    for char in text:
        code = ord(char)
        if 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF:
            return "Chinese"
    return "English"

