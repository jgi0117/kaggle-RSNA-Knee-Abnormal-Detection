import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from knee_guidance import (
    LABEL_COLS,
    KneeGuidanceClassifier,
    build_error_table,
    evaluate_predictions,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
    )
    parser.add_argument(
        "--guidance-dir",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument(
        "--model",
        default="mistralai/Mistral-7B-Instruct-v0.3",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional quick-test row limit.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(args.csv)

    expert_df = df[
        df[LABEL_COLS]
        .notna()
        .all(axis=1)
    ].copy()

    if args.limit is not None:
        expert_df = expert_df.head(
            args.limit
        )

    print(
        f"Expert-labeled studies: "
        f"{len(expert_df)}"
    )

    classifier = KneeGuidanceClassifier(
        guidance_dir=args.guidance_dir,
        model_name=args.model,
    )

    prediction_rows = []
    raw_rows = []

    for idx, row in tqdm(
        expert_df.iterrows(),
        total=len(expert_df),
        desc="Guidance classification",
    ):
        predictions, raw_outputs = (
            classifier.predict_report(
                str(row["Report"])
            )
        )

        prediction_rows.append(
            {
                target: (
                    np.nan
                    if value is None
                    else value
                )
                for target, value
                in predictions.items()
            }
        )

        raw_rows.append(
            {
                "index": idx,
                "raw_outputs": json.dumps(
                    raw_outputs,
                    ensure_ascii=False,
                ),
            }
        )

    pred_df = pd.DataFrame(
        prediction_rows,
        index=expert_df.index,
    )

    metrics, overall_accuracy = (
        evaluate_predictions(
            expert_df,
            pred_df,
        )
    )

    errors = build_error_table(
        expert_df,
        pred_df,
    )

    prediction_output = pd.concat(
        [
            expert_df[
                ["Report"] + LABEL_COLS
            ],
            pred_df.add_prefix("Pred_"),
        ],
        axis=1,
    )

    prediction_output.to_csv(
        output_dir / "predictions.csv",
        index=False,
    )

    metrics.to_csv(
        output_dir / "metrics_by_target.csv",
        index=False,
    )

    errors.to_csv(
        output_dir / "errors.csv",
        index=False,
    )

    pd.DataFrame(raw_rows).to_csv(
        output_dir / "raw_outputs.csv",
        index=False,
    )

    print()
    print(metrics.to_string(index=False))
    print(
        f"\nOverall Accuracy: "
        f"{overall_accuracy:.4f}"
    )
    print(
        f"Saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()
