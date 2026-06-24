import streamlit as st

from config import (
    GENERATED_CONCENTRATIONS,
    REAL_CALIBRATION_CONCENTRATIONS,
    CONCENTRATION_COLORS,
    DEFAULT_END_BV,
    DEFAULT_POINT_COUNT,
)


def make_curve_options():
    options = []
    for conc in REAL_CALIBRATION_CONCENTRATIONS:
        options.append({"type": "Real", "conc": float(conc), "label": f"{conc} mg/L — Real"})
    for conc in GENERATED_CONCENTRATIONS:
        label = f"{conc} mg/L "
        if conc in REAL_CALIBRATION_CONCENTRATIONS:
            label += " (model)"
        options.append({"type": "Generated", "conc": float(conc), "label": label})
    return options


def curve_key_id(curve):
    c = int(curve["conc"]) if float(curve["conc"]).is_integer() else curve["conc"]
    return f"{curve['type']}_{c}"


def format_curve(curve):
    return curve["label"]


def default_color_for_curve(curve):
    conc = int(curve["conc"])
    return CONCENTRATION_COLORS.get(conc, "#333333")


def sidebar_controls(uploaded_file_is_ready=False):
    st.sidebar.header("Settings")
    curve_options = make_curve_options()
    if uploaded_file_is_ready:
        selected_curves = st.sidebar.multiselect(
            "Select curves to compare",
            options=curve_options,
            default=[],
            format_func=format_curve,
            help="Upload Excel first, then choose curves. Real and Generated curves are separate.",
        )
    else:
        selected_curves = []
        st.sidebar.info("Upload the Excel file first, then choose curves.")

    curve_colors = {}
    end_bv_map = {}
    point_count_map = {}

    if selected_curves:
        with st.sidebar.expander("Curve colors", expanded=True):
            st.caption("Each selected curve has its own base color. Cu is darker; Co is lighter.")
            for curve in selected_curves:
                key = curve_key_id(curve)
                curve_colors[key] = st.color_picker(
                    label=curve["label"],
                    value=default_color_for_curve(curve),
                    key=f"color_{key}",
                )

        generated_curves = [c for c in selected_curves if c["type"] == "Generated"]
        if generated_curves:
            with st.sidebar.expander("Generated End BV", expanded=True):
                for curve in generated_curves:
                    conc = int(curve["conc"])
                    default_bv = DEFAULT_END_BV.get(conc, 300)
                    end_bv_map[float(curve["conc"])] = st.number_input(
                        label=f"End BV for {curve['label']}",
                        min_value=20,
                        max_value=1500,
                        value=int(default_bv),
                        step=10,
                        key=f"end_bv_{curve_key_id(curve)}",
                    )
            with st.sidebar.expander("Generated displayed points", expanded=True):
                for curve in generated_curves:
                    conc = int(curve["conc"])
                    default_points = DEFAULT_POINT_COUNT.get(conc, 20)
                    point_count_map[float(curve["conc"])] = st.number_input(
                        label=f"Points for {curve['label']}",
                        min_value=2,
                        max_value=1500,
                        value=int(default_points),
                        step=1,
                        key=f"points_{curve_key_id(curve)}",
                    )

    st.sidebar.header("Model")
    mode = st.sidebar.radio("Generation mode", ["Research", "Smooth"], index=0)
    seed = st.sidebar.number_input("Random seed", min_value=1, max_value=999999, value=42, step=1)
    show_markers = st.sidebar.checkbox("Show markers", value=True)

    if len(selected_curves) > 12:
        st.sidebar.warning("For readable comparison, fewer than 12 curves is usually best.")

    return {
        "selected_curves": selected_curves,
        "curve_colors": curve_colors,
        "end_bv_map": end_bv_map,
        "point_count_map": {float(k): int(v) for k, v in point_count_map.items()},
        "use_gpr": True,
        "gpr_strength": 1.0,
        "show_uncertainty": False,
        "noise_type": "Low",
        "mode": mode,
        "bv_density": "Auto",
        "seed": int(seed),
        "show_markers": show_markers,
    }


def render_header():
    st.title("IX Synthetic Data Generator Pro")
    st.markdown(
        """
        Generate and compare Cu/Co binary ion-exchange breakthrough curves.
        Real and Generated curves for 30, 50, and 500 mg/L can be selected separately.
        """
    )


def render_guidance():
    with st.expander("How to read the plots", expanded=False):
        st.markdown(
            """
            - **30 Real** and **30 Generated** are separate selectable curves.
            - Real curves use Excel data exactly.
            - Generated curves can have End BV and point-count controls.
            - Each selected curve has its own color.
            - Within each curve, Cu is shown as a darker shade and Co as a lighter shade of the selected curve color.
            """
        )
