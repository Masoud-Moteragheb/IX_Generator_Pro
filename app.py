import streamlit as st

from config import AppConfig
from fitting import fit_pipeline
from generator import generate_selected_datasets
from exporter import make_excel, make_combined_csv
from plotting import show_six_charts, show_style_legend
from residual_gpr import ResidualGPRCorrector, SKLEARN_AVAILABLE
from ui import sidebar_controls, render_header, render_guidance


st.set_page_config(
    page_title="IX Generator Pro",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Fitting calibration models...")
def cached_fit_pipeline(uploaded_file_bytes):
    import io
    config = AppConfig()
    return fit_pipeline(io.BytesIO(uploaded_file_bytes), config)


@st.cache_resource(show_spinner="Training residual GPR model...")
def cached_train_gpr(uploaded_file_bytes, seed):
    fit_result = cached_fit_pipeline(uploaded_file_bytes)
    corrector = ResidualGPRCorrector()
    corrector.fit(
        real_data=fit_result["real_data"],
        cu_interps=fit_result["cu_interps"],
        co_interps=fit_result["co_interps"],
        seed=seed,
    )
    return corrector


def main():
    render_header()
    render_guidance()

    uploaded_file = st.file_uploader(
        "Upload calibration Excel file",
        type=["xlsx"],
        help="Required sheets: 30, 50, 500. Required columns: BV, Co, Cu.",
    )

    controls = sidebar_controls(uploaded_file_is_ready=uploaded_file is not None)

    if uploaded_file is None:
        st.info("Upload your calibration Excel file to start.")
        st.stop()

    selected_curves = controls["selected_curves"]
    if not selected_curves:
        st.info("Select one or more curves from the sidebar.")
        st.stop()

    if not SKLEARN_AVAILABLE:
        st.error("scikit-learn is required because GPR is always active. Install it with: pip install scikit-learn")
        st.stop()

    uploaded_file_bytes = uploaded_file.getvalue()

    try:
        fit_result = cached_fit_pipeline(uploaded_file_bytes)
    except Exception as exc:
        st.error(f"Could not read or fit the uploaded workbook: {exc}")
        st.stop()

    try:
        corrector = cached_train_gpr(uploaded_file_bytes, controls["seed"])
    except Exception as exc:
        st.error(f"GPR correction could not be trained: {exc}")
        st.stop()

    datasets = generate_selected_datasets(
        selected_curve_keys=selected_curves,
        real_data=fit_result["real_data"],
        cu_interps=fit_result["cu_interps"],
        co_interps=fit_result["co_interps"],
        noise_type="Low",
        mode=controls["mode"],
        seed=controls["seed"],
        bv_density="Auto",
        end_bv_map=controls["end_bv_map"],
        corrector=corrector,
        use_gpr=True,
        gpr_strength=1.0,
    )

    if not datasets:
        st.warning("No datasets were generated. Please check selected curves.")
        st.stop()

    show_style_legend(selected_curves, curve_color_map=controls["curve_colors"], end_bv_map=controls["end_bv_map"], point_count_map=controls["point_count_map"])
    st.divider()

    show_six_charts(datasets=datasets, show_markers=controls["show_markers"], point_count_map=controls["point_count_map"], curve_color_map=controls["curve_colors"], show_uncertainty=False)
    st.divider()

    st.subheader("Data preview")
    preview_options = sorted(datasets.keys(), key=lambda k: (float(k[1]), k[0]))
    selected_preview = st.selectbox("Preview curve", options=preview_options, format_func=lambda k: f"{int(k[1]) if float(k[1]).is_integer() else k[1]} mg/L - {k[0]}", index=0)
    preview_df = datasets[selected_preview].copy()
    preview_df["BV"] = preview_df["BV"].round(0).astype(int)
    for col in ["Co", "Cu"]:
        preview_df[col] = preview_df[col].where(preview_df[col] >= 1, 0).round(2)
    c0 = float(preview_df["C0"].iloc[0]) if "C0" in preview_df.columns and len(preview_df) else 1.0
    preview_df["Co_C_C0"] = (preview_df["Co"] / c0).round(2)
    preview_df["Cu_C_C0"] = (preview_df["Cu"] / c0).round(2)
    st.dataframe(preview_df[["BV", "Co", "Cu", "Co_C_C0", "Cu_C_C0"]], width="stretch", hide_index=True)

    st.divider()
    excel_file = make_excel(datasets)
    csv_file = make_combined_csv(datasets)
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(label="Download Excel workbook", data=excel_file, file_name="synthetic_IX_results_final_clean.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
    with col_dl2:
        st.download_button(label="Download combined CSV", data=csv_file, file_name="synthetic_IX_results_final_clean.csv", mime="text/csv", width="stretch")


if __name__ == "__main__":
    main()
