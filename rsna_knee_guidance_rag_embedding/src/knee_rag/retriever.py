from pathlib import Path
import json
import torch

from .constants import TARGET_TO_SLUG, TARGET_QUERIES, LABEL_COLS
from .embeddings import E5Embedder


class EmbeddingGuidanceRetriever:
    def __init__(self, index_dir: str, device=None):
        p = Path(index_dir)

        self.records = json.loads(
            (p / "records.json").read_text(
                encoding="utf-8"
            )
        )

        config = json.loads(
            (p / "config.json").read_text(
                encoding="utf-8"
            )
        )

        self.embedding_model = config["embedding_model"]

        self.embeddings = torch.load(
            p / "embeddings.pt",
            map_location="cpu",
        ).float()

        # build target -> candidate row indices
        self.target_indices = {}
        for target, slug in TARGET_TO_SLUG.items():
            self.target_indices[target] = [
                i
                for i, rec in enumerate(self.records)
                if rec["target_slug"] == slug
            ]

        self.embedder = E5Embedder(
            model_name=self.embedding_model,
            device=device,
        )

    def _build_query(self, target: str, report: str):
        # target semantic anchor + patient report
        return (
            f"Target finding: {TARGET_QUERIES[target]}\n\n"
            f"Knee MRI report:\n{report}"
        )

    def retrieve(
        self,
        target: str,
        report: str,
        top_k: int = 2,
    ):
        candidate_idx = self.target_indices[target]

        if not candidate_idx:
            return []

        query = self._build_query(
            target,
            report,
        )

        q = self.embedder.encode(
            [query],
            kind="query",
        )[0]

        sub = self.embeddings[
            candidate_idx
        ]

        # embeddings were L2 normalized at index/query time:
        # dot product == cosine similarity
        scores = sub @ q

        k = min(
            top_k,
            len(candidate_idx),
        )

        values, positions = torch.topk(
            scores,
            k=k,
        )

        results = []

        for score, pos in zip(
            values.tolist(),
            positions.tolist(),
        ):
            rec = dict(
                self.records[
                    candidate_idx[pos]
                ]
            )

            rec["score"] = float(score)
            results.append(rec)

        return results

    def retrieve_all_targets(
        self,
        report: str,
        top_k_per_target: int = 2,
        max_context_chars: int = 16000,
    ):
        gathered = []
        seen = set()

        for target in LABEL_COLS:

            hits = self.retrieve(
                target,
                report,
                top_k=top_k_per_target,
            )

            for rec in hits:

                key = (
                    rec["source"],
                    rec["chunk_id"],
                )

                if key in seen:
                    continue

                seen.add(key)

                gathered.append(
                    (target, rec)
                )

        blocks = []
        total_chars = 0

        for target, rec in gathered:

            block = (
                f"[TARGET: {target}]\n"
                f"[SOURCE: {rec['source']}]\n"
                f"[SIMILARITY: {rec['score']:.4f}]\n"
                f"{rec['text']}"
            )

            if (
                total_chars + len(block)
                > max_context_chars
            ):
                break

            blocks.append(block)
            total_chars += len(block)

        context = "\n\n---\n\n".join(
            blocks
        )

        return context, gathered


# backward-compatible alias
TfidfGuidanceRetriever = EmbeddingGuidanceRetriever
