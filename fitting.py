from typing import Dict
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator

from models import cu_model, co_model
from config import AppConfig


def validate_input_workbook(uploaded_file, config: AppConfig) -> None:
    xls = pd.ExcelFile(uploaded_file)
    missing_sheets = [s for s in config.calibration_sheets if s not in xls.sheet_names]
    if missing_sheets:
        raise ValueError(f"Missing required sheets: {missing_sheets}")

    for sheet in config.calibration_sheets:
        df = pd.read_excel(uploaded_file, sheet_name=sheet)
        missing_cols = [c for c in config.required_columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Sheet {sheet} is missing columns: {missing_cols}")


def load_real_data(uploaded_file, config: AppConfig) -> Dict[float, pd.DataFrame]:
    validate_input_workbook(uploaded_file, config)
    real_data = {}

    for sh in config.calibration_sheets:
        df = pd.read_excel(uploaded_file, sheet_name=sh)
        df = df[list(config.required_columns)].dropna().copy()
        C0 = float(sh)

        df["Co"] = df["Co"].where(df["Co"] >= 1, 0)
        df["Cu"] = df["Cu"].where(df["Cu"] >= 1, 0)

        df["Co_C_C0"] = df["Co"] / C0
        df["Cu_C_C0"] = df["Cu"] / C0
        df["Co_norm"] = df["Co_C_C0"]
        df["Cu_norm"] = df["Cu_C_C0"]
        df["C0"] = C0
        df["DataType"] = "Real"
        real_data[C0] = df.sort_values("BV").reset_index(drop=True)

    return real_data


def _safe_curve_fit(model_func, x, y, p0, bounds, maxfev, fallback):
    try:
        popt, _ = curve_fit(
            model_func,
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=maxfev,
        )
        return popt
    except Exception:
        return np.asarray(fallback, dtype=float)


def fit_calibration_models(real_data):
    params_cu = {}
    params_co = {}

    for C0, df in real_data.items():
        BV = df["BV"].values.astype(float)
        Cu = df["Cu_norm"].values.astype(float)
        Co = df["Co_norm"].values.astype(float)

        bv_median = float(np.median(BV))
        co_peak_guess = float(BV[np.argmax(Co)]) if len(BV) else bv_median

        cu_fallback = [bv_median * 1.2, 25.0, 0.001]
        co_fallback = [bv_median * 0.8, 25.0, 0.35, co_peak_guess, 30.0, 0.001]

        params_cu[C0] = _safe_curve_fit(
            cu_model, BV, Cu,
            p0=cu_fallback,
            bounds=([0, 1, 0], [1500, 400, 0.05]),
            maxfev=30000,
            fallback=cu_fallback,
        )

        params_co[C0] = _safe_curve_fit(
            co_model, BV, Co,
            p0=co_fallback,
            bounds=([0, 1, 0, 0, 1, 0], [1500, 400, 1.8, 1500, 400, 0.10]),
            maxfev=50000,
            fallback=co_fallback,
        )

    return params_cu, params_co


def build_parameter_interpolators(params_cu, params_co):
    known_concs = np.array(sorted(params_cu.keys()), dtype=float)
    log_known = np.log(known_concs)

    cu_matrix = np.array([params_cu[c] for c in known_concs])
    co_matrix = np.array([params_co[c] for c in known_concs])

    cu_interps = [PchipInterpolator(log_known, cu_matrix[:, i]) for i in range(cu_matrix.shape[1])]
    co_interps = [PchipInterpolator(log_known, co_matrix[:, i]) for i in range(co_matrix.shape[1])]

    return cu_interps, co_interps, known_concs


def fit_pipeline(uploaded_file, config: AppConfig):
    real_data = load_real_data(uploaded_file, config)
    params_cu, params_co = fit_calibration_models(real_data)
    cu_interps, co_interps, known_concs = build_parameter_interpolators(params_cu, params_co)

    return {
        "real_data": real_data,
        "params_cu": params_cu,
        "params_co": params_co,
        "cu_interps": cu_interps,
        "co_interps": co_interps,
        "known_concs": known_concs,
    }
