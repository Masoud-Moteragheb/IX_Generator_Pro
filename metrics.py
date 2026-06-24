from typing import Dict, Optional
import numpy as np
import pandas as pd


def first_bv_at_or_above(df: pd.DataFrame, column: str, threshold: float) -> Optional[float]:
    rows = df[df[column] >= threshold]
    if rows.empty:
        return None
    return float(rows.iloc[0]["BV"])


def compute_dataset_metrics(df: pd.DataFrame, key) -> Dict[str, float]:
    data_type, C0 = key
    co_ratio = df["Co_C_C0"]
    cu_ratio = df["Cu_C_C0"]
    co_peak_idx = int(co_ratio.idxmax())
    cu_peak_idx = int(cu_ratio.idxmax())

    return {
        "Data type": data_type,
        "C0": float(C0),
        "End BV": float(df["BV"].max()),
        "Points in data": int(len(df)),
        "Co max/C0": float(co_ratio.max()),
        "Cu max/C0": float(cu_ratio.max()),
        "Co peak BV": float(df.loc[co_peak_idx, "BV"]),
        "Cu peak BV": float(df.loc[cu_peak_idx, "BV"]),
        "Co roll-up": bool(co_ratio.max() > 1.0),
        "Cu roll-up": bool(cu_ratio.max() > 1.0),
        "Co breakthrough BV 5%": first_bv_at_or_above(df, "Co_C_C0", 0.05),
        "Cu breakthrough BV 5%": first_bv_at_or_above(df, "Cu_C_C0", 0.05),
        "Co exhaustion BV 95%": first_bv_at_or_above(df, "Co_C_C0", 0.95),
        "Cu exhaustion BV 95%": first_bv_at_or_above(df, "Cu_C_C0", 0.95),
    }


def build_metrics_table(datasets: Dict[tuple, pd.DataFrame]) -> pd.DataFrame:
    rows = [compute_dataset_metrics(df, key) for key, df in datasets.items()]
    out = pd.DataFrame(rows).sort_values(["C0", "Data type"]).reset_index(drop=True)
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(3)
    return out
