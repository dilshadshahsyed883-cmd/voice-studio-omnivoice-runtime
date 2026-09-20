from app.text_splitter import split_text


def test_preserves_multilingual_text_exactly():
    text = (
        "हिन्दी वाक्य। मराठी वाक्य. ગુજરાતી વાક્ય! "
        "বাংলা বাক্য? هذا نص عربي.\nSecond line."
    ) * 20
    chunks = split_text(text, max_chars=80)
    assert "".join(chunks) == text
    assert all(0 < len(chunk) <= 80 for chunk in chunks)


def test_hard_split_never_loses_characters():
    text = "અ" * 501
    chunks = split_text(text, max_chars=64)
    assert "".join(chunks) == text
    assert max(map(len, chunks)) <= 64
