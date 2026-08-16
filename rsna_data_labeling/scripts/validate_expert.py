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
from knee_guidance.prompting import build_translation_prompt


DEFAULT_TRANSLATIONS_CSV = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "translations.csv"
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
        "--classifier-model",
        default="Qwen/Qwen3-14B",
    )

    parser.add_argument(
        "--translator-model",
        default="Qwen/Qwen3-8B",
    )

    parser.add_argument(
        "--translations-csv",
        default=None,
        help=(
            "Translation CSV override. "
            "If omitted, rsna_data_labeling/data/translations.csv "
            "is reused automatically when present."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional quick-test row limit.",
    )

    return parser.parse_args()


def resolve_translations_csv(cli_path):
    if cli_path is not None:
        path = Path(cli_path)

        if not path.exists():
            raise FileNotFoundError(
                f"translations.csv not found: {path}"
            )

        return path

    if DEFAULT_TRANSLATIONS_CSV.exists():
        return DEFAULT_TRANSLATIONS_CSV

    return None


def load_existing_translations(
    translations_csv,
    expert_df,
):
    translation_df = pd.read_csv(
        translations_csv
    )

    required_cols = {
        "index",
        "translated_report",
    }

    missing_cols = (
        required_cols
        - set(translation_df.columns)
    )

    if missing_cols:
        raise ValueError(
            "translations.csv is missing required columns: "
            f"{sorted(missing_cols)}"
        )

    if translation_df["index"].duplicated().any():
        duplicated = (
            translation_df.loc[
                translation_df["index"].duplicated(
                    keep=False
                ),
                "index",
            ]
            .tolist()
        )

        raise ValueError(
            "translations.csv contains duplicate indices: "
            f"{duplicated[:20]}"
        )

    translation_df = translation_df.set_index(
        "index"
    )

    translated_reports = {}
    translation_rows = []

    for idx, row in expert_df.iterrows():
        if idx not in translation_df.index:
            raise KeyError(
                "translations.csv does not contain "
                f"expert index {idx}"
            )

        value = translation_df.loc[
            idx,
            "translated_report",
        ]

        if pd.isna(value):
            raise ValueError(
                f"translated_report is NaN for index {idx}"
            )

        translated_report = str(value).strip()

        if (
            not translated_report
            or translated_report.lower()
            in {"nan", "none"}
        ):
            raise ValueError(
                "Invalid translated_report for "
                f"index {idx}: {translated_report!r}"
            )

        translated_reports[idx] = translated_report

        translation_rows.append(
            {
                "index": idx,
                "original_report": str(
                    row["Report"]
                ),
                "translated_report": translated_report,
            }
        )

    return (
        translated_reports,
        translation_rows,
    )


def translate_with_qwen(
    expert_df,
    translator_model,
):
    from knee_guidance.llm import QwenTranslator

    translator = QwenTranslator(
        model_name=translator_model,
    )

    translated_reports = {}
    translation_rows = []

    progress = tqdm(
        expert_df.iterrows(),
        total=len(expert_df),
        desc="Translation",
        unit="study",
    )

    for idx, row in progress:
        progress.set_postfix(
            index=idx
        )

        original_report = str(
            row["Report"]
        )

        prompt = build_translation_prompt(
            report=original_report,
        )

        translated_report = translator.translate(
            prompt=prompt,
        )

        translated_reports[idx] = translated_report

        translation_rows.append(
            {
                "index": idx,
                "original_report": original_report,
                "translated_report": translated_report,
            }
        )

    translator.unload()
    del translator

    return (
        translated_reports,
        translation_rows,
    )


def save_checkpoint(
    output_dir,
    prediction_by_index,
    raw_rows,
    failure_rows,
):
    if not prediction_by_index:
        return

    checkpoint_pred_df = (
        pd.DataFrame.from_dict(
            prediction_by_index,
            orient="index",
        )
        .reindex(
            columns=LABEL_COLS
        )
    )

    checkpoint_pred_df.index.name = "index"

    checkpoint_pred_df.reset_index().to_csv(
        output_dir
        / "predictions_checkpoint.csv",
        index=False,
    )

    pd.DataFrame(
        raw_rows
    ).to_csv(
        output_dir
        / "raw_outputs_checkpoint.csv",
        index=False,
    )

    pd.DataFrame(
        failure_rows,
        columns=[
            "index",
            "Target",
            "Error",
        ],
    ).to_csv(
        output_dir
        / "failures_checkpoint.csv",
        index=False,
    )


def main():
    args = parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        args.csv
    )

    expert_df = df[
        df[LABEL_COLS]
        .notna()
        .all(axis=1)
    ].copy()

    if args.limit is not None:
        expert_df = expert_df.head(
            args.limit
        ).copy()

    print(
        f"Expert-labeled studies: "
        f"{len(expert_df)}"
    )

    translations_csv = resolve_translations_csv(
        args.translations_csv
    )

    if translations_csv is not None:
        print(
            "\n[Stage 1/2] "
            f"Using existing translations: {translations_csv}"
        )
        print(
            "Qwen translation skipped."
        )

        (
            translated_reports,
            translation_rows,
        ) = load_existing_translations(
            translations_csv=translations_csv,
            expert_df=expert_df,
        )

    else:
        print(
            "\n[Stage 1/2] "
            "translations.csv not found."
        )
        print(
            "Running Qwen3-8B translation."
        )

        (
            translated_reports,
            translation_rows,
        ) = translate_with_qwen(
            expert_df=expert_df,
            translator_model=args.translator_model,
        )

    pd.DataFrame(
        translation_rows
    ).to_csv(
        output_dir
        / "translations.csv",
        index=False,
    )

    print(
        "\n[Stage 2/2] "
        f"Classifying with {args.classifier_model}"
    )

    classifier = KneeGuidanceClassifier(
        guidance_dir=args.guidance_dir,
        model_name=args.classifier_model,
    )

    prediction_by_index = {}
    raw_rows = []
    failure_rows = []

    progress = tqdm(
        expert_df.iterrows(),
        total=len(expert_df),
        desc="Classification",
        unit="study",
    )

    for idx, _ in progress:
        progress.set_postfix(
            index=idx
        )

        translated_report = (
            translated_reports[idx]
        )

        (
            predictions,
            raw_outputs,
            failures,
        ) = (
            classifier.classify_translated_report(
                translated_report=translated_report,
                show_progress=True,
            )
        )

        prediction_by_index[idx] = {
            target: (
                np.nan
                if value is None
                else value
            )
            for target, value
            in predictions.items()
        }

        raw_rows.append(
            {
                "index": idx,
                "raw_outputs": json.dumps(
                    raw_outputs,
                    ensure_ascii=False,
                ),
            }
        )

        for target, error in failures.items():
            failure_rows.append(
                {
                    "index": idx,
                    "Target": target,
                    "Error": error,
                }
            )

        save_checkpoint(
            output_dir=output_dir,
            prediction_by_index=prediction_by_index,
            raw_rows=raw_rows,
            failure_rows=failure_rows,
        )

    classifier.unload()

    pred_df = (
        pd.DataFrame.from_dict(
            prediction_by_index,
            orient="index",
        )
        .reindex(
            index=expert_df.index,
            columns=LABEL_COLS,
        )
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

    translation_df = pd.DataFrame(
        translation_rows
    ).set_index(
        "index"
    )

    prediction_output = pd.concat(
        [
            expert_df[
                ["Report"] + LABEL_COLS
            ],
            translation_df[
                ["translated_report"]
            ],
            pred_df.add_prefix(
                "Pred_"
            ),
        ],
        axis=1,
    )

    prediction_output.to_csv(
        output_dir
        / "predictions.csv",
        index=False,
    )

    metrics.to_csv(
        output_dir
        / "metrics_by_target.csv",
        index=False,
    )

    errors.to_csv(
        output_dir
        / "errors.csv",
        index=False,
    )

    pd.DataFrame(
        raw_rows
    ).to_csv(
        output_dir
        / "raw_outputs.csv",
        index=False,
    )

    pd.DataFrame(
        failure_rows,
        columns=[
            "index",
            "Target",
            "Error",
        ],
    ).to_csv(
        output_dir
        / "failures.csv",
        index=False,
    )

    print()

    print(
        metrics.to_string(
            index=False
        )
    )

    print(
        f"\nOverall Accuracy: "
        f"{overall_accuracy:.4f}"
    )

    total_decisions = (
        len(expert_df)
        * len(LABEL_COLS)
    )

    print(
        "Parse/generation failures: "
        f"{len(failure_rows)}/{total_decisions}"
    )

    if failure_rows:
        print(
            "WARNING: Accuracy excludes failed/NaN "
            "predictions. Check failures.csv."
        )

    print(
        f"Saved to: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
