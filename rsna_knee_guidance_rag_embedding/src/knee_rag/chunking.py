from pathlib import Path
import re


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_chars: int = 1800, overlap_chars: int = 250):
    """Heading/paragraph-aware character chunker."""
    text = _clean(text)
    if not text:
        return []

    blocks = re.split(r"\n(?=#{1,6}\s)|\n\n+", text)
    chunks, current = [], ""

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        candidate = f"{current}\n\n{block}".strip() if current else block

        if len(candidate) <= chunk_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(block) <= chunk_chars:
            current = block
            continue

        # 긴 단락 fallback
        start = 0
        step = max(1, chunk_chars - overlap_chars)
        while start < len(block):
            piece = block[start:start + chunk_chars].strip()
            if piece:
                chunks.append(piece)
            start += step
        current = ""

    if current:
        chunks.append(current)

    return chunks


def load_guidance_chunks(guidance_dir: str, chunk_chars=1800, overlap_chars=250):
    root = Path(guidance_dir)
    records = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.lower().startswith("readme"):
            continue
        if path.suffix.lower() not in {".md", ".txt"}:
            continue

        rel = path.relative_to(root)
        if len(rel.parts) < 2:
            continue

        target_slug = rel.parts[0]
        text = path.read_text(encoding="utf-8", errors="ignore")

        for i, chunk in enumerate(chunk_text(text, chunk_chars, overlap_chars)):
            records.append({
                "target_slug": target_slug,
                "source": str(rel),
                "chunk_id": i,
                "text": chunk,
            })

    return records
