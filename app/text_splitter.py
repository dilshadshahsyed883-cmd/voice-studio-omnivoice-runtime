from __future__ import annotations

BREAK_CHARS = set(".!?।॥؟\n")


def split_text(text: str, max_chars: int = 420) -> list[str]:
    """Split text without dropping or altering a single input character."""
    if not text:
        return []
    if max_chars < 32:
        raise ValueError("max_chars must be >= 32")

    chunks: list[str] = []
    start = 0
    n = len(text)

    while start < n:
        hard_end = min(start + max_chars, n)
        if hard_end == n:
            chunks.append(text[start:n])
            break

        cut = None
        for i in range(hard_end - 1, start, -1):
            if text[i] in BREAK_CHARS:
                cut = i + 1
                break
        if cut is None:
            for i in range(hard_end - 1, start, -1):
                if text[i].isspace():
                    cut = i + 1
                    break
        if cut is None or cut <= start:
            cut = hard_end

        chunks.append(text[start:cut])
        start = cut

    if "".join(chunks) != text:
        raise AssertionError("text splitter lost or changed input characters")
    if any(len(chunk) > max_chars for chunk in chunks):
        raise AssertionError("text splitter emitted an oversized chunk")
    return chunks
