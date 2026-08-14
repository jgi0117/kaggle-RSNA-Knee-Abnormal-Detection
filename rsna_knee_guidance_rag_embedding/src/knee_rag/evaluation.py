import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .constants import LABEL_COLS


def evaluate_predictions(expert_df: pd.DataFrame, pred_df: pd.DataFrame):
    rows = []

    for label in LABEL_COLS:
        y_true = expert_df[label].astype(int).to_numpy()
        y_pred = pred_df[label].to_numpy(dtype=float)
        valid = ~np.isnan(y_pred)

        yt = y_true[valid]
        yp = y_pred[valid].astype(int)

        rows.append({
            "Target": label,
            "N": int(valid.sum()),
            "Positive": int(yt.sum()),
            "Accuracy": accuracy_score(yt, yp) if len(yt) else np.nan,
            "Precision": precision_score(yt, yp, zero_division=0) if len(yt) else np.nan,
            "Recall": recall_score(yt, yp, zero_division=0) if len(yt) else np.nan,
            "F1": f1_score(yt, yp, zero_division=0) if len(yt) else np.nan,
        })

    metrics = pd.DataFrame(rows)

    true_all = expert_df[LABEL_COLS].astype(int).to_numpy()
    pred_all = pred_df[LABEL_COLS].to_numpy(dtype=float)
    valid_all = ~np.isnan(pred_all)
    overall = float((true_all[valid_all] == pred_all[valid_all]).mean())

    return metrics, overall


def build_error_table(expert_df: pd.DataFrame, pred_df: pd.DataFrame):
    errors = []
    for idx in expert_df.index:
        for label in LABEL_COLS:
            p = pred_df.loc[idx, label]
            if pd.isna(p):
                continue
            t = int(expert_df.loc[idx, label])
            p = int(p)
            if t != p:
                errors.append({
                    "index": idx,
                    "Target": label,
                    "True": t,
                    "Pred": p,
                    "Report": expert_df.loc[idx, "Report"],
                })
    return pd.DataFrame(errors)
