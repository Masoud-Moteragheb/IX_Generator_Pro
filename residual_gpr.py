from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

SKLEARN_AVAILABLE = True
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:
    SKLEARN_AVAILABLE = False
    GaussianProcessRegressor = None
    ConstantKernel = RBF = WhiteKernel = None
    StandardScaler = None

from generator import generate_data


ION_CODE = {"Cu": 0.0, "Co": 1.0}


class ResidualGPRCorrector:
    def __init__(self):
        if not SKLEARN_AVAILABLE:
            raise ModuleNotFoundError("scikit-learn is not installed. Install it with: pip install scikit-learn")
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.model: Optional[GaussianProcessRegressor] = None
        self.is_fitted = False

    def _make_features(self, C0, BV, ion):
        C0 = np.asarray(C0, dtype=float)
        BV = np.asarray(BV, dtype=float)
        ion_code = np.asarray(ion, dtype=float)
        return np.column_stack([np.log(C0), BV, BV / np.maximum(C0, 1e-9), ion_code])

    def fit(self, real_data: Dict[float, pd.DataFrame], cu_interps, co_interps, seed=123):
        X_rows, y_rows = [], []

        for C0, df_exp in real_data.items():
            df_exp = df_exp.sort_values("BV").reset_index(drop=True).copy()
            df_base = generate_data(
                C0=float(C0),
                cu_interps=cu_interps,
                co_interps=co_interps,
                noise_type="Low",
                mode="Smooth",
                seed=seed + int(C0),
                bv_density="Dense",
                end_bv=float(df_exp["BV"].max()),
            ).sort_values("BV").reset_index(drop=True)

            BV_exp = df_exp["BV"].values.astype(float)
            base_cu = np.interp(BV_exp, df_base["BV"].values, df_base["Cu_C_C0"].values)
            base_co = np.interp(BV_exp, df_base["BV"].values, df_base["Co_C_C0"].values)

            exp_cu = df_exp["Cu_norm"].values.astype(float)
            exp_co = df_exp["Co_norm"].values.astype(float)

            res_cu = exp_cu - base_cu
            res_co = exp_co - base_co

            X_rows.append(self._make_features(np.full_like(BV_exp, float(C0)), BV_exp, np.full_like(BV_exp, ION_CODE["Cu"])))
            X_rows.append(self._make_features(np.full_like(BV_exp, float(C0)), BV_exp, np.full_like(BV_exp, ION_CODE["Co"])))
            y_rows.append(res_cu.reshape(-1, 1))
            y_rows.append(res_co.reshape(-1, 1))

        X = np.vstack(X_rows)
        y = np.vstack(y_rows)

        X_scaled = self.x_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y).ravel()

        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2))
            * RBF(length_scale=np.ones(X_scaled.shape[1]), length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
        )

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=False,
            n_restarts_optimizer=3,
            random_state=seed,
        )
        self.model.fit(X_scaled, y_scaled)
        self.is_fitted = True
        return self

    def predict_residual(self, C0: float, BV: np.ndarray, ion_name: str) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("ResidualGPRCorrector is not fitted.")
        X = self._make_features(
            C0=np.full_like(BV, float(C0), dtype=float),
            BV=BV.astype(float),
            ion=np.full_like(BV, ION_CODE[ion_name], dtype=float),
        )
        X_scaled = self.x_scaler.transform(X)
        y_scaled_mean, y_scaled_std = self.model.predict(X_scaled, return_std=True)
        mean = self.y_scaler.inverse_transform(y_scaled_mean.reshape(-1, 1)).ravel()
        std = y_scaled_std * float(self.y_scaler.scale_[0])
        return mean, std


def apply_residual_gpr_correction(df_base, C0, corrector, strength=1.0, uncertainty_multiplier=1.96):
    df = df_base.copy()
    BV = df["BV"].values.astype(float)

    cu_res, cu_std = corrector.predict_residual(C0, BV, "Cu")
    co_res, co_std = corrector.predict_residual(C0, BV, "Co")

    df["Cu_base"] = df["Cu"]
    df["Co_base"] = df["Co"]
    df["Cu_C_C0_base"] = df["Cu_C_C0"]
    df["Co_C_C0_base"] = df["Co_C_C0"]

    cu_norm = np.clip(df["Cu_C_C0"].values + strength * cu_res, 0.0, 1.02)
    co_norm = np.clip(df["Co_C_C0"].values + strength * co_res, 0.0, 1.72)

    df["Cu_C_C0"] = cu_norm
    df["Co_C_C0"] = co_norm
    cu_conc = cu_norm * C0
    co_conc = co_norm * C0
    cu_conc = np.where(cu_conc < 1, 0, cu_conc)
    co_conc = np.where(co_conc < 1, 0, co_conc)
    df["Cu"] = np.round(cu_conc, 3)
    df["Co"] = np.round(co_conc, 3)
    df["Cu_C_C0"] = df["Cu"] / C0
    df["Co_C_C0"] = df["Co"] / C0

    df["Cu_residual_gpr"] = cu_res
    df["Co_residual_gpr"] = co_res
    df["Cu_gpr_std"] = cu_std
    df["Co_gpr_std"] = co_std

    df["Cu_lower_95"] = np.clip((cu_norm - uncertainty_multiplier * cu_std) * C0, 0, C0 * 1.02)
    df["Cu_upper_95"] = np.clip((cu_norm + uncertainty_multiplier * cu_std) * C0, 0, C0 * 1.02)
    df["Co_lower_95"] = np.clip((co_norm - uncertainty_multiplier * co_std) * C0, 0, C0 * 1.72)
    df["Co_upper_95"] = np.clip((co_norm + uncertainty_multiplier * co_std) * C0, 0, C0 * 1.72)

    return df
