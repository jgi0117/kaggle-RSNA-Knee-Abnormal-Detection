import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from .constants import LABEL_COLS


def evaluate_predictions(
    expert_df: pd.DataFrame,
    pred_df: pd.DataFrame,
):
    rows = []

    for target in LABEL_COLS:
        y_true = expert_df[target].astype(int).to_numpy()
        y_pred = pd.to_numeric(
            pred_df[target],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid = ~np.isnan(y_pred)

        yt = y_true[valid]
        yp = y_pred[valid].astype(int)

        rows.append(
            {
                "Target": target,
                "N": int(valid.sum()),
                "Positive": int(yt.sum()) if len(yt) else 0,
                "Accuracy": (
                    accuracy_score(yt, yp)
                    if len(yt)
                    else np.nan
                ),
                "Precision": (
                    precision_score(
                        yt,
                        yp,
                        zero_division=0,
                    )
                    if len(yt)
                    else np.nan
                ),
                "Recall": (
                    recall_score(
                        yt,
                        yp,
                        zero_division=0,
                    )
                    if len(yt)
                    else np.nan
                ),
                "F1": (
                    f1_score(
                        yt,
                        yp,
                        zero_division=0,
                    )
                    if len(yt)
                    else np.nan
                ),
            }
        )

    metrics = pd.DataFrame(rows)

    true_all = expert_df[
        LABEL_COLS
    ].astype(int).to_numpy()

    pred_all = pred_df[
        LABEL_COLS
    ].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)

    valid_all = ~np.isnan(pred_all)

    if valid_all.sum() == 0:
        overall_accuracy = np.nan
    else:
        overall_accuracy = float(
            (
                true_all[valid_all]
                == pred_all[valid_all]
            ).mean()
        )

    return metrics, overall_accuracy


def build_error_table(
    expert_df: pd.DataFrame,
    pred_df: pd.DataFrame,
):
    errors = []

    for idx in expert_df.index:
        for target in LABEL_COLS:
            pred = pd.to_numeric(
                pd.Series([pred_df.loc[idx, target]]),
                errors="coerce",
            ).iloc[0]

            if pd.isna(pred):
                continue

            true = int(expert_df.loc[idx, target])
            pred = int(pred)

            if true != pred:
                errors.append(
                    {
                        "index": idx,
                        "Target": target,
                        "True": true,
                        "Pred": pred,
                        "Report": expert_df.loc[
                            idx,
                            "Report",
                        ],
                    }
                )

    return pd.DataFrame(errors)
