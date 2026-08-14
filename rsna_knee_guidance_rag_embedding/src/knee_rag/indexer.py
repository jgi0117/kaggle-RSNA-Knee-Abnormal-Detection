from pathlib import Path
import json
import torch

from .chunking import load_guidance_chunks
from .embeddings import E5Embedder


def build_index(
    guidance_dir: str,
    output_dir: str,
    embedding_model="intfloat/multilingual-e5-small",
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    records = load_guidance_chunks(guidance_dir)

    if not records:
        raise ValueError(
            "No guidance chunks found. Add .md/.txt files under "
            "guidance/<target_slug>/ first."
        )

    embedder = E5Embedder(
        model_name=embedding_model,
    )

    corpus = [r["text"] for r in records]

    embeddings = embedder.encode(
        corpus,
        kind="passage",
    )

    torch.save(
        embeddings,
        output / "embeddings.pt",
    )

    (output / "records.json").write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (output / "config.json").write_text(
        json.dumps(
            {
                "embedding_model": embedding_model,
                "num_chunks": len(records),
                "embedding_dim": embeddings.shape[1],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return len(records)
