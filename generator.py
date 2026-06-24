
from typing import Iterable, Literal, Tuple, Dict, Optional
import numpy as np
import pandas as pd

from models import cu_model, co_model

NoiseType = Literal["Low", "Medium", "High"]
ModeType = Literal["Research", "Smooth", "Experimental", "Publication"]

def interpolate_params(C0, cu_interps, co_interps) -> Tuple[np.ndarray, np.ndarray]:
    logC = np.log(float(C0))
    cu_p = np.array([float(f(logC)) for f in cu_interps], dtype=float)
    co_p = np.array([float(f(logC)) for f in co_interps], dtype=float)

    cu_p[1] = max(cu_p[1], 1.0)
    cu_p[2] = np.clip(cu_p[2], 0.0, 0.05)

    co_p[1] = max(co_p[1], 1.0)
    co_p[2] = np.clip(co_p[2], 0.0, 1.8)
    co_p[4] = max(co_p[4], 1.0)
    co_p[5] = np.clip(co_p[5], 0.0, 0.10)

    return cu_p, co_p

def generate_BV_grid(C0, cu_p, co_p, density: str = "Auto", end_bv: Optional[float] = None) -> np.ndarray:
    if end_bv is None:
        max_point = max(float(cu_p[0]), float(co_p[0]), float(co_p[3]) + 2.5 * float(co_p[4]))
        bv_end = max(max_point + 80.0, 80.0)
    else:
        bv_end = max(float(end_bv), 10.0)

    if density == "Dense":
        step = 2
    elif density == "Medium":
        step = 4
    elif density == "Sparse":
        step = 8
    else:
        step = 3 if C0 >= 300 else 5

    BV = np.arange(0, bv_end + step, step, dtype=float)
    BV = np.unique(np.concatenate([[0, 1, 5], BV, [bv_end]]))
    BV = BV[BV <= bv_end]
    return np.sort(BV)

def noise_parameters(noise_type: NoiseType, mode: ModeType, C0: float):
    if mode == "Smooth":
        return 0.0, 0.0
    if mode == "Publication":
        return 0.010, 0.0008 * C0
    if noise_type == "Low":
        return 0.015, 0.0010 * C0
    if noise_type == "Medium":
        return 0.035, 0.0030 * C0
    return 0.055, 0.0050 * C0

def add_noise(y, C0, noise_type: NoiseType, mode: ModeType, rng: np.random.Generator):
    rel_noise, abs_noise = noise_parameters(noise_type, mode, C0)
    if rel_noise == 0 and abs_noise == 0:
        return np.maximum(y, 0)
    sigma = rel_noise * np.maximum(y, 0.02 * C0) + abs_noise
    noise = rng.normal(0, sigma)
    return np.maximum(y + noise, 0)

def smooth_publication_curve(y: np.ndarray) -> np.ndarray:
    if len(y) < 5:
        return y
    return pd.Series(y).rolling(window=3, center=True, min_periods=1).mean().values

def generate_data(
    C0: float,
    cu_interps,
    co_interps,
    noise_type: NoiseType = "Medium",
    mode: ModeType = "Research",
    seed: int = 42,
    bv_density: str = "Auto",
    end_bv: Optional[float] = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cu_p, co_p = interpolate_params(C0, cu_interps, co_interps)
    BV = generate_BV_grid(C0, cu_p, co_p, density=bv_density, end_bv=end_bv)

    Cu = cu_model(BV, *cu_p) * C0
    Co = co_model(BV, *co_p) * C0

    Cu = np.minimum(Cu, C0 * 1.01)
    Co = np.minimum(Co, C0 * 1.70)

    Cu = add_noise(Cu, C0, noise_type, mode, rng)
    Co = add_noise(Co, C0, noise_type, mode, rng)

    if mode == "Publication":
        Cu = smooth_publication_curve(Cu)
        Co = smooth_publication_curve(Co)

    Cu = np.minimum(Cu, C0 * 1.02)
    Co = np.minimum(Co, C0 * 1.72)

    Cu = np.where(Cu < 1, 0, Cu)
    Co = np.where(Co < 1, 0, Co)

    df = pd.DataFrame({"BV": BV, "Co": np.round(Co, 3), "Cu": np.round(Cu, 3)})
    df["Co_C_C0"] = df["Co"] / C0
    df["Cu_C_C0"] = df["Cu"] / C0
    df["C0"] = float(C0)
    df["DataType"] = "Generated"
    return df

def generate_multiple_datasets_hybrid(
    concentrations: Iterable[float],
    cu_interps,
    co_interps,
    noise_type: NoiseType,
    mode: ModeType,
    seed: int,
    bv_density: str,
    end_bv_map: Optional[Dict[float, float]] = None,
    corrector=None,
    use_gpr: bool = False,
    gpr_strength: float = 1.0,
):
    from residual_gpr import apply_residual_gpr_correction

    datasets = {}
    for conc in concentrations:
        conc = float(conc)
        end_bv = None
        if end_bv_map:
            end_bv = end_bv_map.get(float(conc), end_bv_map.get(int(conc), None))

        df_base = generate_data(
            C0=conc,
            cu_interps=cu_interps,
            co_interps=co_interps,
            noise_type=noise_type,
            mode=mode,
            seed=seed + int(round(conc * 10)),
            bv_density=bv_density,
            end_bv=end_bv,
        )

        if use_gpr and corrector is not None:
            df_out = apply_residual_gpr_correction(df_base, conc, corrector, strength=gpr_strength)
            df_out["DataType"] = "Generated"
        else:
            df_out = df_base

        datasets[("Generated", conc)] = df_out

    return datasets

def add_real_datasets(generated_datasets, real_data, selected_concs):
    out = dict(generated_datasets)
    selected_set = {float(c) for c in selected_concs}
    for C0, df in real_data.items():
        if float(C0) in selected_set:
            real_df = df.copy()
            real_df["DataType"] = "Real"
            out[("Real", float(C0))] = real_df
    return out

def generate_selected_datasets(
    selected_curve_keys, real_data, cu_interps, co_interps, noise_type, mode, seed, bv_density,
    end_bv_map=None, corrector=None, use_gpr=False, gpr_strength=1.0):
    from residual_gpr import apply_residual_gpr_correction
    datasets = {}
    for curve in selected_curve_keys:
        data_type = curve["type"]
        conc = float(curve["conc"])
        if data_type == "Real":
            if conc in real_data:
                df = real_data[conc].copy()
                df["DataType"] = "Real"
                datasets[("Real", conc)] = df
            continue
        end_bv = None
        if end_bv_map:
            end_bv = end_bv_map.get(float(conc), end_bv_map.get(int(conc), None))
        df_base = generate_data(C0=conc, cu_interps=cu_interps, co_interps=co_interps, noise_type=noise_type, mode=mode, seed=seed + int(round(conc * 10)), bv_density=bv_density, end_bv=end_bv)
        if use_gpr and corrector is not None:
            df_out = apply_residual_gpr_correction(df_base, conc, corrector, strength=gpr_strength)
            df_out["DataType"] = "Generated"
        else:
            df_out = df_base
        datasets[("Generated", conc)] = df_out
    return datasets