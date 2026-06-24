from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class AppConfig:
    calibration_sheets: tuple = ("30", "50", "500")
    required_columns: tuple = ("BV", "Co", "Cu")


REAL_CALIBRATION_CONCENTRATIONS = [30, 50, 500]

GENERATED_CONCENTRATIONS: List[int] = [
    30, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500
]

CONCENTRATION_COLORS: Dict[int, str] = {
    30: "#1f77b4",
    50: "#274CFF",
    75: "#2ca02c",
    100: "#E31A1C",
    150: "#9467bd",
    200: "#39C20D",
    250: "#e377c2",
    300: "#E6DF00",
    350: "#bcbd22",
    400: "#20BFD3",
    450: "#003f5c",
    500: "#FF00F0",
}


def default_end_bv_for_concentration(concentration: float) -> int:
    """
    Automatic default End BV based on feed concentration.

    Low concentration curves need longer BV ranges.
    High concentration curves exhaust earlier and need shorter BV ranges.

    Approximate defaults:
    <100 mg/L -> near 1200 BV
    400-500 mg/L -> near 200 BV
    """
    c = float(concentration)

    if c < 100:
        return 1200
    if c >= 500:
        return 200
    if c >= 400:
        return 220

    # Smooth decrease between 100 and 400 mg/L:
    # 100 -> 900, 400 -> 220
    if c >= 100:
        value = 900 - (c - 100) * (680 / 300)
        return int(round(value / 10) * 10)

    return 1200


DEFAULT_END_BV: Dict[int, int] = {
    c: default_end_bv_for_concentration(c)
    for c in GENERATED_CONCENTRATIONS
}

# Default displayed/generated points for every generated concentration
DEFAULT_GENERATED_POINT_COUNT = 20

DEFAULT_POINT_COUNT: Dict[int, int] = {
    c: DEFAULT_GENERATED_POINT_COUNT
    for c in GENERATED_CONCENTRATIONS
}

ION_MARKER_SYMBOL = {
    "Cu": "diamond",
    "Co": "circle",
}
