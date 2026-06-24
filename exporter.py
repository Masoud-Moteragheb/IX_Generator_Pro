from io import BytesIO
from typing import Dict
import pandas as pd


def _sheet_name(key):
    data_type, C0 = key
    c = str(int(C0)) if float(C0).is_integer() else str(C0)
    prefix = "Real" if data_type == "Real" else "Gen"
    return f"{prefix}_{c}"[:31]


def _five_column_export(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "BV": df["BV"].round(0).astype(int),
        "Co": df["Co"],
        "Cu": df["Cu"],
        "Co_C_C0": df["Co_C_C0"],
        "Cu_C_C0": df["Cu_C_C0"],
    })
    out["Co"] = out["Co"].where(out["Co"] >= 1, 0)
    out["Cu"] = out["Cu"].where(out["Cu"] >= 1, 0)
    if "C0" in df.columns and len(df["C0"]) > 0:
        c0 = float(df["C0"].iloc[0])
        out["Co_C_C0"] = out["Co"] / c0
        out["Cu_C_C0"] = out["Cu"] / c0
    for col in ["Co", "Cu", "Co_C_C0", "Cu_C_C0"]:
        out[col] = out[col].round(2)
    return out


def make_excel(datasets: Dict[tuple, pd.DataFrame], metrics_df: pd.DataFrame = None) -> BytesIO:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for key, df in datasets.items():
            _five_column_export(df).to_excel(writer, sheet_name=_sheet_name(key), index=False)
    output.seek(0)
    return output


def make_combined_csv(datasets: Dict[tuple, pd.DataFrame]) -> bytes:
    combined = []
    for key, df in datasets.items():
        data_type, C0 = key
        temp = _five_column_export(df)
        temp["C0"] = C0
        temp["DataType"] = data_type
        combined.append(temp)
    return pd.concat(combined, ignore_index=True).to_csv(index=False).encode("utf-8")
