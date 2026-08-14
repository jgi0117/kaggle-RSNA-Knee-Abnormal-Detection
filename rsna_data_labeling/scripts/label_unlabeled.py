import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from knee_guidance import (
    LABEL_COLS,
    KneeGuidanceClassifier,
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
        "--output",
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
    )

    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.csv)

    unlabeled_df = df[
        df[LABEL_COLS]
        .isna()
        .any(axis=1)
    ].copy()

    if args.limit is not None:
        unlabeled_df = unlabeled_df.head(
            args.limit
        )

    print(
        f"Reports to pseudo-label: "
        f"{len(unlabeled_df)}"
    )

    classifier = KneeGuidanceClassifier(
        guidance_dir=args.guidance_dir,
        model_name=args.model,
        load_in_4bit=not args.no_4bit,
    )

    output_rows = []

    for idx, row in tqdm(
        unlabeled_df.iterrows(),
        total=len(unlabeled_df),
        desc="Pseudo-labeling",
    ):
        predictions, _ = (
            classifier.predict_report(
                str(row["Report"])
            )
        )

        result = {
            "source_index": idx,
            "Report": row["Report"],
        }

        for target in LABEL_COLS:
            value = predictions[target]

            result[target] = (
                np.nan
                if value is None
                else value
            )

        output_rows.append(result)

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        output_rows
    ).to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()
