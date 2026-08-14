import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from knee_rag.constants import LABEL_COLS
from knee_rag.pipeline import KneeGuidanceRAG
from knee_rag.evaluation import evaluate_predictions, build_error_table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--index-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--no-4bit", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    expert = df[df[LABEL_COLS].notna().all(axis=1)].copy()
    print("Expert rows:", len(expert))

    rag = KneeGuidanceRAG(
        index_dir=args.index_dir,
        model_name=args.model,
        load_in_4bit=not args.no_4bit,
        top_k_per_target=args.top_k,
    )

    preds, raws = [], []
    for _, row in tqdm(expert.iterrows(), total=len(expert), desc="RAG validation"):
        result = rag.predict(str(row["Report"]))
        pred = result["prediction"]

        if pred is None:
            preds.append({label: np.nan for label in LABEL_COLS})
        else:
            preds.append(pred)
        raws.append(result["raw_output"])

    pred_df = pd.DataFrame(preds, index=expert.index)
    metrics, overall = evaluate_predictions(expert, pred_df)
    errors = build_error_table(expert, pred_df)

    prediction_out = pd.concat(
        [
            expert[["Report"] + LABEL_COLS],
            pred_df.add_prefix("Pred_"),
            pd.Series(raws, index=expert.index, name="RawOutput"),
        ],
        axis=1,
    )

    prediction_out.to_csv(out / "predictions.csv", index=False)
    metrics.to_csv(out / "metrics_by_target.csv", index=False)
    errors.to_csv(out / "errors.csv", index=False)

    print(metrics.to_string(index=False))
    print(f"\nOverall Accuracy: {overall:.4f}")
    print("Saved:", out)


if __name__ == "__main__":
    main()
