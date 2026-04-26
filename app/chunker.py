from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 150
) -> List[str]:
    """
    Recursive character-based chunking that respects natural text boundaries.
    Priority order: paragraphs -> sentences -> words -> characters.
    """

    if not text or not text.strip():
        return []

    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text: str, separators: List[str]) -> List[str]:
        if not separators:
            chunks = []
            for i in range(0, len(text), chunk_size - overlap):
                chunks.append(text[i:i + chunk_size])
            return chunks

        sep = separators[0]
        splits = text.split(sep) if sep else list(text)

        chunks = []
        current = ""

        for split in splits:
            candidate = current + (sep if current else "") + split

            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())

                if len(split) > chunk_size:
                    sub_chunks = _split(split, separators[1:])
                    chunks.extend(sub_chunks[:-1])
                    current = sub_chunks[-1] if sub_chunks else ""
                else:
                    overlap_text = current[-overlap:] if overlap and current else ""
                    current = overlap_text + (sep if overlap_text else "") + split

        if current.strip():
            chunks.append(current.strip())

        return chunks

    raw_chunks = _split(text, separators)
    # Filter noise: remove very short chunks (page headers, stray numbers)
    cleaned = [c for c in raw_chunks if len(c.split()) > 10]
    return cleaned
