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
        "--translator-model",
        default="Qwen/Qwen3-8B",
    )

    parser.add_argument(
        "--classifier-model",
        default="google/medgemma-1.5-4b-it",
    )

    parser.add_argument(
        "--translations-csv",
        default=None,
        help=(
            "Optional translation CSV override. "
            "If omitted, rsna_data_labeling/data/translations.csv "
            "is used automatically when it exists."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional quick-test row limit.",
    )

    return parser.parse_args()


def resolve_translations_csv(
    cli_path,
):
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

        translated_report = str(
            translation_df.loc[
                idx,
                "translated_report",
            ]
        ).strip()

        if not translated_report:
            raise ValueError(
                "Empty translated_report for "
                f"index {idx}"
            )

        translated_reports[idx] = (
            translated_report
        )

        translation_rows.append(
            {
                "index": idx,
                "original_report": str(
                    row["Report"]
                ),
                "translated_report": (
                    translated_report
                ),
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
    # Import lazily so Qwen is never loaded when
    # data/translations.csv is already available.
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

        translated_report = (
            translator.translate(
                prompt=prompt,
            )
        )

        translated_reports[idx] = (
            translated_report
        )

        translation_rows.append(
            {
                "index": idx,
                "original_report": (
                    original_report
                ),
                "translated_report": (
                    translated_report
                ),
            }
        )

    translator.unload()
    del translator

    return (
        translated_reports,
        translation_rows,
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

    # =====================================================
    # Load data
    # =====================================================
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

    # =====================================================
    # Stage 1: Translation
    # =====================================================
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
            "Running Qwen translation."
        )

        (
            translated_reports,
            translation_rows,
        ) = translate_with_qwen(
            expert_df=expert_df,
            translator_model=args.translator_model,
        )

    # Always save the exact translations used.
    pd.DataFrame(
        translation_rows
    ).to_csv(
        output_dir
        / "translations.csv",
        index=False,
    )

    # =====================================================
    # Stage 2: MedGemma classification
    # =====================================================
    print(
        "\n[Stage 2/2] "
        "Classifying with MedGemma"
    )

    classifier = KneeGuidanceClassifier(
        guidance_dir=args.guidance_dir,
        model_name=args.classifier_model,
    )

    prediction_rows = []
    raw_rows = []

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
        ) = (
            classifier.classify_translated_report(
                translated_report=translated_report,
                show_progress=True,
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

    # =====================================================
    # Prediction DataFrame
    # =====================================================
    pred_df = pd.DataFrame(
        prediction_rows,
        index=expert_df.index,
    )

    # =====================================================
    # Evaluation
    # =====================================================
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

    # =====================================================
    # Final prediction output
    # =====================================================
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

    # =====================================================
    # Save
    # =====================================================
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

    # =====================================================
    # Results
    # =====================================================
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

    print(
        f"Saved to: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
