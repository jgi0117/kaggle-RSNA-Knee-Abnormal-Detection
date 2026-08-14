from pathlib import Path
import sys

import pandas as pd
from transformers import AutoTokenizer


# =========================================================
# Paths
# =========================================================
DATA_DIR = Path(__file__).resolve().parent
BASE_DIR = DATA_DIR.parent

TRAIN_CSV_PATH = DATA_DIR / "train.csv"
GUIDANCE_DIR = BASE_DIR / "guidance"
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(SRC_DIR))


# =========================================================
# Project imports
# =========================================================
from knee_guidance.constants import LABEL_COLS
from knee_guidance.guidance import GuidanceStore
from knee_guidance.prompting import build_target_prompt


# =========================================================
# Model
# =========================================================
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"


# =========================================================
# Load train.csv
# =========================================================
train_df = pd.read_csv(TRAIN_CSV_PATH)

expert_df = train_df[
    train_df[LABEL_COLS].notna().all(axis=1)
].copy()

print(f"Expert-labeled studies: {len(expert_df)}")


# =========================================================
# Load tokenizer / guidance
# =========================================================
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

guidance_store = GuidanceStore(
    GUIDANCE_DIR
)


# =========================================================
# Count prompt tokens
# =========================================================
rows = []

for idx, row in expert_df.iterrows():

    report = str(row["Report"])

    for target in LABEL_COLS:

        guidance = guidance_store.get(
            target
        )

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

        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        input_ids = tokenizer(
            chat_text,
            add_special_tokens=False,
        )["input_ids"]

        num_tokens = len(input_ids)

        rows.append(
            {
                "index": idx,
                "target": target,
                "tokens": num_tokens,
            }
        )


token_df = pd.DataFrame(rows)


# =========================================================
# Overall statistics
# =========================================================
print("\n=== Overall Token Statistics ===")

print(
    token_df["tokens"].describe()
)

print()
print(
    "Over 4096 :",
    (token_df["tokens"] > 4096).sum()
)

print(
    "Over 8192 :",
    (token_df["tokens"] > 8192).sum()
)

print(
    "Over 16384:",
    (token_df["tokens"] > 16384).sum()
)

print(
    "Over 32768:",
    (token_df["tokens"] > 32768).sum()
)

print(
    "Max tokens:",
    token_df["tokens"].max()
)


# =========================================================
# Statistics by target
# =========================================================
print("\n=== Token Statistics by Target ===")

target_stats = (
    token_df
    .groupby("target")["tokens"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max",
        ]
    )
    .sort_values(
        "max",
        ascending=False,
    )
)

print(target_stats)


# =========================================================
# Longest prompts
# =========================================================
print("\n=== Top 20 Longest Prompts ===")

print(
    token_df
    .sort_values(
        "tokens",
        ascending=False,
    )
    .head(20)
    .to_string(index=False)
)


# =========================================================
# Save
# =========================================================
OUTPUT_PATH = (
    DATA_DIR
    / "prompt_token_lengths.csv"
)

token_df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print(
    f"\nSaved: {OUTPUT_PATH}"
)