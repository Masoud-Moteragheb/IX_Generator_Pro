from typing import Dict, Literal, Tuple, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config import CONCENTRATION_COLORS, ION_MARKER_SYMBOL

PlotMode = Literal["concentration", "normalized"]


def _curve_key_id(data_type, C0):
    c = int(C0) if float(C0).is_integer() else C0
    return f"{data_type}_{c}"


def _fallback_color(C0):
    key = int(C0) if float(C0).is_integer() else C0
    return CONCENTRATION_COLORS.get(key, "#333333")


def _base_color_for_curve(data_type, C0, curve_color_map=None):
    key = _curve_key_id(data_type, C0)
    if curve_color_map and key in curve_color_map:
        return curve_color_map[key]
    return _fallback_color(C0)


def _hex_to_rgb(hex_color):
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return (80, 80, 80)
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(
        *[int(max(0, min(255, x))) for x in rgb]
    )


def _shade_color(hex_color: str, ion: str):
    if ion == "Cu":
        return hex_color

    r, g, b = _hex_to_rgb(hex_color)
    lighten = 0.45
    rgb = (
        r + (255 - r) * lighten,
        g + (255 - g) * lighten,
        b + (255 - b) * lighten,
    )
    return _rgb_to_hex(rgb)


def _hex_to_rgba(hex_color, alpha):
    r, g, b = _hex_to_rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"


def _y_column(ion, mode):
    return ion if mode == "concentration" else f"{ion}_C_C0"


def _trace_name(ion: str, data_type: str, C0: float, mode: PlotMode) -> str:
    C0_label = int(C0) if float(C0).is_integer() else C0
    ion_label = ion if mode == "concentration" else f"{ion}/C0"

    if data_type == "Real":
        return f"{ion_label} | {C0_label} mg/L | Real"

    return f"{ion_label} | {C0_label} mg/L"


def _point_count_for_concentration(C0, point_count_map, default=20):
    if point_count_map:
        if float(C0) in point_count_map:
            return int(point_count_map[float(C0)])
        if int(C0) in point_count_map:
            return int(point_count_map[int(C0)])
    return int(default)


def _sample_generated_points(df, n_points):
    if df.empty:
        return df

    df = df.sort_values("BV").reset_index(drop=True).copy()
    n = max(int(n_points), 2)
    target = np.linspace(float(df["BV"].min()), float(df["BV"].max()), n)

    out = pd.DataFrame({"BV": target})

    for col in [
        "Co", "Cu",
        "Co_C_C0", "Cu_C_C0",
        "Co_lower_95", "Co_upper_95",
        "Cu_lower_95", "Cu_upper_95",
    ]:
        if col in df.columns:
            out[col] = np.interp(target, df["BV"].values, df[col].values)

    return out


def _add_uncertainty_band(fig, df_plot, ion, data_type, C0, color, mode):
    if mode != "concentration" or data_type != "Generated":
        return

    lo, hi = f"{ion}_lower_95", f"{ion}_upper_95"

    if lo not in df_plot.columns or hi not in df_plot.columns:
        return

    fig.add_trace(
        go.Scatter(
            x=df_plot["BV"],
            y=df_plot[hi],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            legendgroup=f"{data_type}-{C0}-{ion}",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_plot["BV"],
            y=df_plot[lo],
            mode="lines",
            fill="tonexty",
            fillcolor=_hex_to_rgba(color, 0.12),
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            legendgroup=f"{data_type}-{C0}-{ion}",
        )
    )


def _title_html(title):
    st.markdown(
        f"""
        <div style="
            text-align:center;
            font-weight:700;
            font-size:20px;
            margin-top:18px;
            margin-bottom:4px;
        ">
            {title}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _dash_for_data_type(data_type):
    return "dash" if data_type == "Real" else "solid"


def make_multi_chart(
    datasets: Dict[tuple, pd.DataFrame],
    mode: PlotMode,
    ions: Tuple[str, ...],
    y_title: str,
    hline=None,
    hline_text=None,
    show_markers=True,
    point_count_map=None,
    chart_height=560,
    curve_color_map=None,
    show_uncertainty=False,
):
    fig = go.Figure()

    for key in sorted(
        datasets.keys(),
        key=lambda k: (float(k[1]), 0 if k[0] == "Generated" else 1),
    ):
        data_type, C0 = key
        df_full = datasets[key]

        if data_type == "Generated":
            df_plot = _sample_generated_points(
                df_full,
                _point_count_for_concentration(C0, point_count_map, 20),
            )
        else:
            df_plot = df_full.sort_values("BV").reset_index(drop=True).copy()

        base = _base_color_for_curve(data_type, C0, curve_color_map)

        for ion in ions:
            color = _shade_color(base, ion)
            y_col = _y_column(ion, mode)
            name = _trace_name(ion, data_type, C0, mode)

            if show_uncertainty:
                _add_uncertainty_band(fig, df_plot, ion, data_type, C0, color, mode)

            fig.add_trace(
                go.Scatter(
                    x=df_plot["BV"],
                    y=df_plot[y_col],
                    mode="lines+markers" if show_markers else "lines",
                    name=name,
                    legendgroup=f"{data_type}-{C0}-{ion}",
                    showlegend=True,
                    line=dict(
                        color=color,
                        dash=_dash_for_data_type(data_type),
                        width=2.9 if ion == "Cu" else 2.5,
                    ),
                    marker=dict(
                        color=color,
                        size=7 if data_type == "Generated" else 6,
                        symbol=ION_MARKER_SYMBOL.get(ion, "circle"),
                        line=dict(width=0.8, color="white"),
                    ),
                    opacity=0.95 if data_type == "Generated" else 0.78,
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "BV=%{x:.2f}<br>"
                        + y_title +
                        "=%{y:.4g}<extra></extra>"
                    ),
                )
            )

    if hline is not None:
        fig.add_hline(
            y=hline,
            line_dash="dot",
            line_width=1.8,
            line_color="black",
            annotation_text=hline_text,
            annotation_position="top left",
        )

    fig.update_layout(
        title=dict(text=""),
        template="plotly_white",
        height=chart_height,
        hovermode="x unified",
        margin=dict(l=70, r=120, t=20, b=65),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(0,0,0,0.18)",
            borderwidth=1,
            font=dict(size=10),
        ),
        xaxis=dict(
            title="BV",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
            linecolor="rgba(0,0,0,0.35)",
            mirror=True,
        ),
        yaxis=dict(
            title=y_title,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
            linecolor="rgba(0,0,0,0.35)",
            mirror=True,
        ),
        font=dict(family="Arial", size=13),
    )

    return fig


def prepare_publication_figure(fig):
    fig_export = go.Figure(fig)

    fig_export.update_layout(
        template="plotly_white",
        width=1800,
        height=1100,
        margin=dict(l=110, r=260, t=60, b=110),
        font=dict(
            family="Arial",
            size=20,
            color="black",
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="white",
            bordercolor="#BFBFBF",
            borderwidth=1,
            font=dict(
                family="Arial",
                size=17,
                color="black",
            ),
        ),
        xaxis=dict(
            title_font=dict(size=24, color="black"),
            tickfont=dict(size=18, color="black"),
            showgrid=True,
            gridcolor="#D9D9D9",
            gridwidth=1,
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            zeroline=False,
        ),
        yaxis=dict(
            title_font=dict(size=24, color="black"),
            tickfont=dict(size=18, color="black"),
            showgrid=True,
            gridcolor="#D9D9D9",
            gridwidth=1,
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            zeroline=False,
        ),
    )

    for trace in fig_export.data:
        if hasattr(trace, "line") and trace.line is not None:
            if trace.line.width is None or trace.line.width > 0:
                trace.line.width = 4.0

        if hasattr(trace, "marker") and trace.marker is not None:
            trace.marker.size = 9
            trace.marker.line = dict(width=1.2, color="white")

    return fig_export


def export_chart_buttons(fig, filename):
    safe_filename = filename.replace(" ", "_").replace("/", "_")
    png_key = f"{safe_filename}_png"

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "Generate PNG",
            key=f"generate_png_{safe_filename}",
            width="stretch",
        ):
            try:
                with st.spinner("Generating PNG..."):
                    fig_export = prepare_publication_figure(fig)
                    st.session_state[png_key] = fig_export.to_image(
                        format="png",
                        width=1800,
                        height=1100,
                        scale=2,
                    )
            except Exception as e:
                import traceback
                st.error("PNG export failed")
                st.code(traceback.format_exc())

    with col2:
        if png_key in st.session_state:
            st.download_button(
                label="Download PNG",
                data=st.session_state[png_key],
                file_name=f"{safe_filename}.png",
                mime="image/png",
                width="stretch",
                key=f"download_png_{safe_filename}",
            )
        else:
            st.download_button(
                label="Download PNG",
                data=b"",
                file_name=f"{safe_filename}.png",
                mime="image/png",
                width="stretch",
                disabled=True,
                key=f"download_png_disabled_{safe_filename}",
            )


def _show_chart(fig, title):
    _title_html(title)

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "toImageButtonOptions": {
                "format": "png",
                "filename": title.replace(" ", "_").replace("/", "_"),
                "width": 1600,
                "height": 950,
                "scale": 2,
            }
        },
    )

    export_chart_buttons(fig, filename=title)


def show_six_charts(
    datasets,
    show_markers,
    point_count_map=None,
    curve_color_map=None,
    show_uncertainty=False,
):
    st.subheader("1–3: Concentration vs BV")

    _show_chart(
        make_multi_chart(
            datasets,
            "concentration",
            ("Cu", "Co"),
            "Concentration, mg/L",
            show_markers=show_markers,
            point_count_map=point_count_map,
            chart_height=620,
            curve_color_map=curve_color_map,
            show_uncertainty=show_uncertainty,
        ),
        "Cu and Co concentration vs BV",
    )

    _show_chart(
        make_multi_chart(
            datasets,
            "concentration",
            ("Cu",),
            "Cu, mg/L",
            show_markers=show_markers,
            point_count_map=point_count_map,
            chart_height=570,
            curve_color_map=curve_color_map,
            show_uncertainty=show_uncertainty,
        ),
        "Cu concentration vs BV",
    )

    _show_chart(
        make_multi_chart(
            datasets,
            "concentration",
            ("Co",),
            "Co, mg/L",
            show_markers=show_markers,
            point_count_map=point_count_map,
            chart_height=570,
            curve_color_map=curve_color_map,
            show_uncertainty=show_uncertainty,
        ),
        "Co concentration vs BV",
    )

    st.subheader("4–6: Normalized concentration C/C0 vs BV")

    _show_chart(
        make_multi_chart(
            datasets,
            "normalized",
            ("Cu", "Co"),
            "C/C0",
            hline=1,
            hline_text="C/C0 = 1",
            show_markers=show_markers,
            point_count_map=point_count_map,
            chart_height=620,
            curve_color_map=curve_color_map,
        ),
        "Cu/C0 and Co/C0 vs BV",
    )

    _show_chart(
        make_multi_chart(
            datasets,
            "normalized",
            ("Cu",),
            "Cu/C0",
            hline=1,
            hline_text="C/C0 = 1",
            show_markers=show_markers,
            point_count_map=point_count_map,
            chart_height=570,
            curve_color_map=curve_color_map,
        ),
        "Cu/C0 vs BV",
    )

    _show_chart(
        make_multi_chart(
            datasets,
            "normalized",
            ("Co",),
            "Co/C0",
            hline=1,
            hline_text="C/C0 = 1",
            show_markers=show_markers,
            point_count_map=point_count_map,
            chart_height=570,
            curve_color_map=curve_color_map,
        ),
        "Co/C0 vs BV",
    )


def show_style_legend(
    selected_curves,
    curve_color_map=None,
    end_bv_map=None,
    point_count_map=None,
):
    st.markdown("#### Curve settings key")

    cols = st.columns(min(len(selected_curves), 5))

    for i, curve in enumerate(selected_curves):
        dt = curve["type"]
        C0 = float(curve["conc"])
        key = _curve_key_id(dt, C0)

        base = (
            curve_color_map.get(key, _fallback_color(C0))
            if curve_color_map
            else _fallback_color(C0)
        )

        cu = _shade_color(base, "Cu")
        co = _shade_color(base, "Co")

        label = int(C0) if float(C0).is_integer() else C0

        if dt == "Generated":
            eb = end_bv_map.get(float(C0)) if end_bv_map else None
            pc = point_count_map.get(float(C0)) if point_count_map else None

            controls = (
                f"<br><small>End BV: {eb:.0f} | Points: {pc}</small>"
                if eb is not None and pc is not None
                else ""
            )
        else:
            controls = "<br><small>Excel points only</small>"

        with cols[i % len(cols)]:
            st.markdown(
                f"""
                <div style="margin-bottom:8px;">
                    <div style="display:flex;align-items:center;gap:8px;">
                        <div style="
                            width:14px;
                            height:14px;
                            border-radius:50%;
                            background:{cu};
                            border:1px solid #444;
                        "></div>
                        <div style="
                            width:14px;
                            height:14px;
                            border-radius:50%;
                            background:{co};
                            border:1px solid #444;
                        "></div>
                        <span>{label} mg/L — {dt}</span>
                    </div>
                    {controls}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(
        "For each curve, Cu uses the selected color and Co uses a lighter shade."
    )