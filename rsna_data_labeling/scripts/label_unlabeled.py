import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from knee_guidance import LABEL_COLS, KneeGuidanceClassifier
from knee_guidance.prompting import build_translation_prompt

DEFAULT_TRANSLATIONS_CSV = Path(__file__).resolve().parents[1] / "data" / "translations.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--guidance-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--translator-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--classifier-model", default="google/medgemma-1.5-4b-it")
    parser.add_argument("--translations-csv", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def resolve_translations_csv(cli_path):
    if cli_path is not None:
        path = Path(cli_path)
        if not path.exists():
            raise FileNotFoundError(f"translations.csv not found: {path}")
        return path
    return DEFAULT_TRANSLATIONS_CSV if DEFAULT_TRANSLATIONS_CSV.exists() else None


def load_translation_cache(path):
    if path is None:
        return {}
    df = pd.read_csv(path)
    required_cols = {"index", "translated_report"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"translations.csv is missing required columns: {sorted(missing)}")
    if df["index"].duplicated().any():
        raise ValueError("translations.csv contains duplicate index values.")
    cache = {}
    for _, row in df.iterrows():
        value = row["translated_report"]
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            continue
        cache[int(row["index"])] = text
    return cache


def translate_missing_reports(unlabeled_df, translation_cache, translator_model):
    missing_indices = [idx for idx in unlabeled_df.index if idx not in translation_cache]
    if not missing_indices:
        print("All translations found. Qwen translation skipped.")
        return translation_cache
    print(f"Missing translations: {len(missing_indices)}")
    print("Loading Qwen only for missing reports.")
    from knee_guidance.llm import QwenTranslator
    translator = QwenTranslator(model_name=translator_model)
    for idx in tqdm(missing_indices, desc="Translation", unit="study"):
        original_report = str(unlabeled_df.loc[idx, "Report"])
        prompt = build_translation_prompt(report=original_report)
        translation_cache[idx] = translator.translate(prompt=prompt)
    translator.unload()
    del translator
    return translation_cache


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    unlabeled_df = df[df[LABEL_COLS].isna().any(axis=1)].copy()
    if args.limit is not None:
        unlabeled_df = unlabeled_df.head(args.limit).copy()
    print(f"Reports to pseudo-label: {len(unlabeled_df)}")

    translations_path = resolve_translations_csv(args.translations_csv)
    translation_cache = load_translation_cache(translations_path)
    translation_cache = translate_missing_reports(
        unlabeled_df=unlabeled_df,
        translation_cache=translation_cache,
        translator_model=args.translator_model,
    )

    classifier = KneeGuidanceClassifier(
        guidance_dir=args.guidance_dir,
        model_name=args.classifier_model,
    )

    output_rows = []
    failure_rows = []
    for idx, row in tqdm(unlabeled_df.iterrows(), total=len(unlabeled_df), desc="Pseudo-labeling", unit="study"):
        translated_report = translation_cache[idx]
        predictions, _, failures = classifier.classify_translated_report(
            translated_report=translated_report,
            show_progress=False,
        )
        result = {
            "source_index": idx,
            "Report": row["Report"],
            "translated_report": translated_report,
        }
        for target in LABEL_COLS:
            value = predictions[target]
            result[target] = np.nan if value is None else value
        output_rows.append(result)
        for target, error in failures.items():
            failure_rows.append({"index": idx, "Target": target, "Error": error})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(output_path, index=False)
    failures_path = output_path.with_name(output_path.stem + "_failures.csv")
    pd.DataFrame(failure_rows, columns=["index", "Target", "Error"]).to_csv(failures_path, index=False)
    print(f"Saved: {output_path}")
    print(f"Failures: {len(failure_rows)}")


if __name__ == "__main__":
    main()
