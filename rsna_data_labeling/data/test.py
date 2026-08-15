from pathlib import Path
import sys

import pandas as pd
from transformers import AutoTokenizer


DATA_DIR = Path(__file__).resolve().parent
BASE_DIR = DATA_DIR.parent

TRAIN_CSV_PATH = DATA_DIR / "train.csv"
GUIDANCE_DIR = BASE_DIR / "guidance"
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from knee_guidance.constants import LABEL_COLS
from knee_guidance.guidance import GuidanceStore
from knee_guidance.prompting import build_target_prompt


MODEL_NAME = "Qwen/Qwen3-8B"


def main():
    df = pd.read_csv(TRAIN_CSV_PATH)

    expert_df = df[
        df[LABEL_COLS]
        .notna()
        .all(axis=1)
    ].copy()

    print(f"Expert studies: {len(expert_df)}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    guidance_store = GuidanceStore(
        GUIDANCE_DIR
    )

    results = []

    for index, row in expert_df.iterrows():
        report = str(row["Report"])

        for target in LABEL_COLS:
            guidance = guidance_store.get(
                target
            )

            # 실제 classifier에 들어가는
            # guidance + common prompt + report 전체
            prompt = build_target_prompt(
                report=report,
                target=target,
                guidance=guidance,
            )

            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

            # 실제 QwenTargetClassifier._generate()와 동일
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            tokens = tokenizer(
                formatted_text,
                add_special_tokens=False,
            )["input_ids"]

            results.append(
                {
                    "index": index,
                    "target": target,
                    "report_chars": len(report),
                    "guidance_chars": len(guidance),
                    "prompt_chars": len(prompt),
                    "input_tokens": len(tokens),
                }
            )

    result_df = pd.DataFrame(results)

    print("\n=== Token statistics ===")
    print(
        result_df["input_tokens"]
        .describe()
        .to_string()
    )

    print("\n=== Max token input ===")
    max_row = result_df.loc[
        result_df["input_tokens"].idxmax()
    ]
    print(max_row.to_string())

    print("\n=== Max tokens by target ===")
    print(
        result_df
        .groupby("target")["input_tokens"]
        .agg(["mean", "max"])
        .sort_values("max", ascending=False)
        .to_string()
    )


if __name__ == "__main__":
    main()