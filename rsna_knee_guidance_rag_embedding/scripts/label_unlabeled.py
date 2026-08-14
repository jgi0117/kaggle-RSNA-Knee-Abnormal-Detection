import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from knee_rag.constants import LABEL_COLS
from knee_rag.pipeline import KneeGuidanceRAG


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--index-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--no-4bit", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    unlabeled = df[df[LABEL_COLS].isna().any(axis=1)].copy()
    print("Rows to pseudo-label:", len(unlabeled))

    rag = KneeGuidanceRAG(
        index_dir=args.index_dir,
        model_name=args.model,
        load_in_4bit=not args.no_4bit,
        top_k_per_target=args.top_k,
    )

    rows = []
    for idx, row in tqdm(unlabeled.iterrows(), total=len(unlabeled), desc="Pseudo-labeling"):
        result = rag.predict(str(row["Report"]))
        pred = result["prediction"] or {label: np.nan for label in LABEL_COLS}

        record = {"source_index": idx, "Report": row["Report"]}
        record.update(pred)
        rows.append(record)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Saved:", out)


if __name__ == "__main__":
    main()
