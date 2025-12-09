"""
Plotting functions for SPE fitting and charge resolution.

This module provides Plotly-based interative plots for use
in HTML reports, including:

- Fitted SPE parameter distributions
- Visualisations of those paramters mapped to the camera geometry
- Per-pixel SPE histograms and fitted PDF overlays
- Charge-resolution curves
- Charge resolution scaled to CTAO requirements
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import yaml
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from tqdm import tqdm

from sstcam_pixel_analysis.processing import calc_charge_res
from sstcam_pixel_analysis.utilities import (
    fmt,
    format_hz,
    get_pixel_info,
    get_source,
    requirement,
)

SCRIPT_DIR = Path(__file__).resolve().parent

with open(f"{SCRIPT_DIR}/config.yaml", "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

PDE = _cfg["PDE"]

X_POISS = 10 ** np.arange(np.log10(0.3), np.log10(700), 0.1)
Y_POISS = 1 / np.sqrt(X_POISS / PDE)
X_REQ = X_POISS


def plot_charge_res(df):
    """
    Plot relative charge resolution vs expected charge

    Args:
        df (pandas DataFrame): DF of extracted charges

    Returns:
        fig: Plotly HTML figure
    """
    fig = go.Figure()

    unique_nsbs = sorted(df["nsb"].unique())
    n_nsbs = len(unique_nsbs)
    if n_nsbs == 1:
        cmap = sample_colorscale("Plasma", [0])
    else:
        cmap = sample_colorscale(
            "Plasma", [0.9 * i / (n_nsbs - 1) for i in range(n_nsbs)]
        )

    for color, nsb_val in zip(cmap, unique_nsbs):
        y_req = requirement(X_REQ / PDE, nsb=PDE * nsb_val / 1e9, pde=PDE)
        fig.add_trace(
            go.Scatter(
                x=X_REQ,
                y=y_req,
                mode="lines",
                line={"color": color, "dash": "dash"},
                name=f"Requirement (NSB: {format_hz(nsb_val)})",
                hoverinfo="name",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=X_POISS,
            y=Y_POISS,
            mode="lines",
            line={"color": "grey", "dash": "dash"},
            name="Poisson",
            hoverinfo="name",
        )
    )

    for color, nsb_val in zip(cmap, unique_nsbs):
        sub = df[df["nsb"] == nsb_val]
        fig.add_trace(
            go.Scatter(
                x=sub["expected_pe"],
                y=sub["charge_res"],
                mode="markers",
                marker={
                    "size": 9,
                    "color": color,
                },
                name=f"Charge res. (NSB: {format_hz(nsb_val)})",
                hovertemplate=(
                    f"NSB: {format_hz(nsb_val)}<br>"
                    "expected p.e.: %{x}<br>"
                    "Resolution: %{y}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        width=800,
        height=500,
        plot_bgcolor="white",
        xaxis={
            "type": "log",
            "title": "Average Expected p.e.",
            "showgrid": True,
            "zeroline": False,
            "linecolor": "black",
            "mirror": True,
            "range": [np.log10(0.3), np.log10(700)],
        },
        yaxis={
            "type": "log",
            "title": "Fractional charge resolution",
            "showgrid": True,
            "zeroline": False,
            "linecolor": "black",
            "mirror": True,
        },
        legend={
            "x": 1.02,
            "y": 1,
            "xanchor": "left",
            "yanchor": "top",
            "bordercolor": "black",
        },
        margin={"l": 60, "r": 20, "t": 20, "b": 60},
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def plot_charge_res_relative(df_2d):
    """Plot charge resolution divided by the CTAO requirement for each NSB level.

    This shows how the camera's charge resolution compares to the
    CTAO requirement curve, both for individual pixels (hidden by default)
    and the whole camera, for each tested NSB level.

    Args:
        df_2d (pandas.DataFrame): Table of extracted and expected charges
            containing columns: 'pixel_id', 'extracted_pe', 'expected_pe', 'nsb'.

    Returns:
        str: HTML snippet containing the Plotly figure.
    """

    def y_relative(x, y, nsb_val):
        return y / requirement(x / PDE, nsb=PDE * nsb_val / 1e9, pde=PDE)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=X_REQ,
            y=[1] * len(X_POISS),
            mode="lines",
            line={"color": "black", "dash": "dash"},
            name="Requirement",
            hoverinfo="name",
        )
    )

    unique_nsbs = sorted(df_2d["nsb"].unique())
    n_nsbs = len(unique_nsbs)
    if n_nsbs == 1:
        cmap = sample_colorscale("Plasma", [0])
    else:
        cmap = sample_colorscale(
            "Plasma", [0.9 * i / (n_nsbs - 1) for i in range(n_nsbs)]
        )

    for color, nsb_val in zip(cmap, unique_nsbs):

        fig.add_trace(
            go.Scatter(
                x=X_POISS,
                y=y_relative(X_POISS, Y_POISS, nsb_val),
                mode="lines",
                line={"color": color, "dash": "dash"},
                name=f"Poisson (NSB: {format_hz(nsb_val)})",
                hoverinfo="name",
            )
        )

    for color, nsb_val in zip(cmap, unique_nsbs):

        sub_nsb = df_2d[df_2d["nsb"] == nsb_val]

        legend_name = "Individual pixels"

        unique_pix = sorted(sub_nsb["pixel_id"].unique())
        for pix_id in unique_pix:
            sub_pix = sub_nsb[sub_nsb["pixel_id"] == pix_id]
            xax = sorted(sub_pix["expected_pe"].unique())
            yax = [
                calc_charge_res(
                    sub_pix[sub_pix["expected_pe"] == pe]["extracted_pe"], pe
                )
                for pe in xax
            ]

            show_legend = bool(
                (pix_id == unique_pix[0]) and (nsb_val == unique_nsbs[0])
            )

            rgba_color = color.replace("rgb(", "rgba(").replace(
                ")", f",{np.sqrt(1/len(unique_pix))})"
            )
            fig.add_trace(
                go.Scatter(
                    x=xax,
                    y=y_relative(np.array(xax), np.array(yax), nsb_val),
                    mode="lines",
                    line={"color": rgba_color},
                    legendgroup="pixels",
                    showlegend=show_legend,
                    name=legend_name if pix_id == unique_pix[0] else None,
                    visible="legendonly",
                    hovertemplate=(
                        f"Pixel: {pix_id}<br>"
                        "expected p.e.: %{x}<br>"
                        "Rel. Resolution: %{y}<extra></extra>"
                    ),
                )
            )

        x_total = sorted(sub_nsb["expected_pe"].unique())
        y_total = []
        for pe in x_total:
            sub_pe = sub_nsb[sub_nsb["expected_pe"] == pe]
            y_total.append(calc_charge_res(sub_pe["extracted_pe"], pe))

        fig.add_trace(
            go.Scatter(
                x=x_total,
                y=y_relative(np.array(x_total), np.array(y_total), nsb_val),
                mode="markers",
                marker={
                    "size": 9,
                    "color": color,
                },
                name=f"Charge res. (NSB: {format_hz(nsb_val)})",
                hovertemplate=(
                    f"NSB: {format_hz(nsb_val)}<br>"
                    "expected p.e.: %{x}<br>"
                    "Rel. Resolution: %{y}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        width=800,
        height=500,
        plot_bgcolor="white",
        showlegend=True,
        xaxis={
            "type": "log",
            "title": "Average Expected p.e.",
            "showgrid": True,
            "zeroline": False,
            "linecolor": "black",
            "mirror": True,
            "range": [np.log10(0.3), np.log10(700)],
        },
        yaxis={
            "title": "Fractional charge resolution / Requirement",
            "showgrid": True,
            "zeroline": False,
            "linecolor": "black",
            "mirror": True,
            "range": [0, 1.5],
        },
        legend={
            "x": 1.02,
            "y": 1,
            "xanchor": "left",
            "yanchor": "top",
            "bordercolor": "black",
        },
        margin={"l": 60, "r": 20, "t": 20, "b": 60},
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def plot_dispersion(df_2d):
    """
    Plot 2D histogram of expected vs extracted charges

    Args:
        df_2d (pandas DataFrame): DF of all extracted charges

    Returns:
        fig: Plotly HTML figure
    """
    figures = []

    for nsb in tqdm(sorted(df_2d["nsb"].unique()), desc="Saving plots"):
        sub = df_2d[(df_2d["nsb"] == nsb) & (df_2d["extracted_pe"] > 0)]

        x = sub["expected_pe"].values
        y = sub["extracted_pe"].values / sub["expected_pe"].values

        mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        xbins = np.logspace(np.log10(x.min()), np.log10(x.max() * 1.2), 120)
        ybins = np.logspace(-4, 2.05, 500)

        hist, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ybins])
        z = np.log10(hist.T + 1)

        heatmap = go.Heatmap(
            z=z,
            x=xedges[:-1],
            y=yedges[:-1],
            colorscale="Viridis",
            colorbar={"title": "log10(Counts)"},
        )

        ref_line = go.Scatter(
            x=[x.min(), x.max()],
            y=[1, 1],
            mode="lines",
            line={"color": "white", "dash": "dot"},
            name="Extracted = Expected",
        )

        one_line = go.Scatter(
            x=xbins,
            y=1 / xbins,
            mode="lines",
            line={"color": "grey", "dash": "dot"},
            name="Extracted = 1",
        )

        mean_points = (
            sub.groupby("expected_pe")[
                ["extracted_pe", "expected_pe"]
            ]  # <-- Select the columns
            .apply(
                lambda g: (
                    g["expected_pe"].iloc[0],
                    (g["extracted_pe"] / g["expected_pe"]).mean(),
                )
            )
            .tolist()
        )
        mean_x, mean_y = zip(*mean_points)

        mean_line = go.Scatter(
            x=mean_x,
            y=mean_y,
            mode="lines",
            line={"color": "red", "width": 2},
            name="Mean",
        )

        fig = go.Figure([heatmap, ref_line, one_line, mean_line])

        fig.update_layout(
            width=800,
            height=600,
            title=f"NSB: {format_hz(nsb)}",
            plot_bgcolor="white",
            xaxis={"title": "Expected PE", "type": "log"},
            yaxis={"title": "Extracted PE / Expected PE", "type": "log"},
            margin={"l": 60, "r": 20, "t": 40, "b": 60},
            legend={
                "x": 0.99,
                "y": 0.99,
                "xanchor": "right",
                "yanchor": "top",
                "bgcolor": "rgba(255,255,255,0.7)",
            },
        )

        fig.update_traces(hoverinfo="skip")  # disables hover globally

        figures.append(fig.to_html(full_html=False, include_plotlyjs=False))

    return figures


def get_param_text(ipix, illum_no, include_pix, fit):
    """
    Generates the text to go under the per-pixel SPE plots

    Args:
        ipix (int): Pixel index/no.
        illum_no (int): Illumination/file number
        include_pix (bool array): Mask of included pixels for this plot
        fit (object): Class containing fitter and related information

    Returns:
        (str): Monospaced text showing fit parameters
    """
    fitter = fit.fitter

    values = fitter.pixel_values[ipix]
    errors = fitter.pixel_errors[ipix]
    scores = fitter.pixel_scores[ipix]

    keys = [
        f"eped_{illum_no}",
        "eped_sigma",
        "pe",
        "pe_sigma",
        "opct",
        f"lambda_{illum_no}",
        "reduced_chi2",
        "p_value",
    ]

    max_name_len = max(len(k) for k in keys)
    val_err_strings = {}

    for param in keys:
        if param == "p_value":
            sf = 4
        elif param == "chi2":
            sf = 1
        else:
            sf = 2
        if param in values.keys():
            val_str = fmt(values[param], sf)
            if values[param] >= 0 or param == "p_value":
                val_str = " " + val_str
            err_str = fmt(errors[param], sf)
            val_err = f"{val_str} ± {err_str}"
            val_err_strings[param] = val_err
        else:
            val_str = fmt(scores[param], sf)
            if scores[param] >= 0:
                val_str = " " + val_str
            val_err_strings[param] = val_str

    max_val_err_len = max(len(s) for s in val_err_strings.values())

    header_line = f"{'Param':<{max_name_len+4}}{'Value':<{max_val_err_len+1}}vs Mean"
    param_lines = [header_line]

    for param in keys:
        name_str = f"{param:<{max_name_len}}"
        val_err_str = f"{val_err_strings[param]:<{max_val_err_len}}"
        delta_str = ""

        data = np.array(fit.value_lists[param])[include_pix]
        mean = np.mean(data)
        std = np.std(data)
        if param in values.keys():
            delta = values[param] - mean
        else:
            delta = scores[param] - mean
        z_score = abs(delta / std) if std != 0 else 0

        sgn = "+" if delta >= 0 else "-"
        delta_str = f"({sgn}{fmt(abs(delta))})" + "*" * int(z_score)

        param_lines.append(f"{name_str} = {val_err_str}  {delta_str}")

    param_lines.append(
        f"{"peak ratio":<{max_name_len}} =  {fmt(fit.peak_valley[illum_no][ipix],3)}"
    )
    param_lines.append(
        f"{"good fit":<{max_name_len}} =  {fit.good_fit_masks[illum_no][ipix]}"
    )

    return "\n".join(param_lines)


def plot_value_lists_plotly(value_lists, illum_no, include_pix):
    """
    Plot the parameter histograms

    Args:
        value_lists (dict): Dict of arrays of extracted parameters
        illum_no (int): Illumination/file number
        include_pix (bool array): Mask of included pixels for this plot

    Returns:
        fig: Plotly HTML figure
    """
    keys = [
        f"eped_{illum_no}",
        "eped_sigma",
        "pe",
        "pe_sigma",
        "opct",
        f"lambda_{illum_no}",
        "reduced_chi2",
        "p_value",
    ]
    rows, cols = 5, 2

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=keys,
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
    )

    for idx, key in enumerate(keys):
        row = idx // cols + 1
        col = idx % cols + 1

        data = np.array(value_lists[key])[include_pix]
        mean_val = np.mean(data)
        std_val = np.std(data)

        mn, mx = min(data), max(data)
        sz = (mx - mn) / 70

        fig.add_trace(
            go.Histogram(
                x=data,
                xbins={"start": mn, "end": mx, "size": sz},
                marker={"color": "dodgerblue"},
            ),
            row=row,
            col=col,
        )

        fig.add_vline(
            x=mean_val, line_dash="dash", line_color="lightgray", row=row, col=col
        )

        fig.add_annotation(
            text=f"mean={fmt(mean_val)}<br> std dev={fmt(std_val)}",
            showarrow=False,
            borderwidth=0,
            row=row,
            col=col,
            x=max(data),
            xanchor="right",
            yanchor="bottom",
            y=0,
            bgcolor="rgba(255,255,255,0.8)",
        )

    fig.update_layout(
        height=1000,
        width=700,
        showlegend=False,
        margin={"t": 50, "b": 40, "l": 40, "r": 20},
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def plot_all_fits_plotly(pixel_nos, fitter, illum_no, include_pix):
    """
    Plot the SPE histograms and their coresponding fits for every pixel.

    Args:
        pixel_nos (array): List of pixel IDs
        fitter (Fitter): The SPE fitter
        illum_no (int): Illumination/file number
        include_pix (bool array): Mask of included pixels for this plot

    Returns:
        fig: Plotly HTML figure
    """
    pixel_nos = np.array(pixel_nos)[include_pix]
    ipix_filtered = np.where(include_pix)[0]

    slot_asic_groups = defaultdict(list)
    slot_asic_indices = defaultdict(list)

    for idx, pixel_no in enumerate(pixel_nos):
        slot, asic, _ = get_pixel_info(pixel_no)
        key = (slot, asic)
        slot_asic_groups[key].append(pixel_no)
        slot_asic_indices[key].append(idx)

    group_keys = sorted(slot_asic_groups.keys())
    num_groups = len(group_keys)
    num_rows = int(np.ceil(num_groups / 2))

    subplot_titles = [f"Slot {slot}, ASIC {asic}" for slot, asic in group_keys]

    fig = make_subplots(
        rows=num_rows,
        cols=2,
        subplot_titles=subplot_titles,
        vertical_spacing=0.15 / (num_rows - 1) if num_rows > 1 else 0,
    )

    for i, (slot, asic) in enumerate(group_keys):
        row = i // 2 + 1
        col = i % 2 + 1
        for idx in slot_asic_indices[(slot, asic)]:
            pixel_no = pixel_nos[idx]
            ipix = ipix_filtered[idx]
            fit_x = fitter.pixel_arrays[ipix][illum_no]["fit_x"]
            fit_y = fitter.pixel_arrays[ipix][illum_no]["fit_y"]

            slot, asic, asic_ch = get_pixel_info(pixel_no)

            label = (f"Pixel ID: {pixel_no}<br>Slot: {slot}<br>"
                     f"ASIC: {asic}<br>ASIC Ch: {asic_ch}<extra></extra>")

            fig.add_trace(
                go.Scatter(
                    x=fit_x,
                    y=fit_y,
                    mode="lines",
                    hovertemplate=label,
                    line={"width": 2},
                ),
                row=row,
                col=col,
            )

    fig.update_yaxes(autorangeoptions={"include": 0, "clipmin": 0})

    fig.update_layout(
        showlegend=False,
        width=800,
        height=250 * num_rows,
        margin={"l": 30, "r": 30, "t": 30, "b": 30},
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def plot_fit_plotly(c, fit, param_text, ipix, pixel_no, illum_no, hist_bins):
    """Plot a pixel's SPE histogram with its fitted PDF.

    Generates a histogram of extracted charges, overlays the fitted SPE model,
    and returns both the HTML plot and the formatted parameter text.

    Args:
        c (array): Extracted ADC count values for this pixel.
        fit (FitResults): Object containing fitter, hist range, and parameter lists.
        param_text (str): Text describing fitted parameters.
        ipix (int): Index of the pixel within the fitted-pixels array.
        pixel_no (int): Physical pixel ID.
        illum_no (int): Illumination/file index.
        hist_bins (int): Number of histogram bins.

    Returns:
        tuple:
            pixel_plot (str): HTML of the Plotly plot.
            pixel_text (str): Parameter text formatted for HTML.
    """

    slot, asic, asic_ch = get_pixel_info(pixel_no)
    pixel_no = str(pixel_no)

    hist, bin_edges = np.histogram(
        c, bins=hist_bins, range=fit.hist_range, density=True
    )
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fit_x = fit.fitter.pixel_arrays[ipix][illum_no]["fit_x"]
    fit_y = fit.fitter.pixel_arrays[ipix][illum_no]["fit_y"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=hist,
            mode="lines",
            name="Data",
            line={"color": "grey", "shape": "hvh"},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=fit_x,
            y=fit_y,
            mode="lines",
            name="Fit",
            line={"color": "maroon"},
        )
    )

    fig.update_yaxes(autorangeoptions={"include": 0, "clipmin": 0})

    fig.update_layout(
        width=600,
        height=350,
        title=f"Pixel ID: {pixel_no}, Slot: {slot}, ASIC: {asic}, ASIC Ch: {asic_ch}",
        margin={"t": 50, "b": 10},
        xaxis_title="Integrated charge (mV)",
        yaxis_title="Counts (normalised)",
        legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
    )

    pixel_plot = fig.to_html(full_html=False, include_plotlyjs=False)
    pixel_text = param_text.replace("\n", "<br>")

    return pixel_plot, pixel_text


def get_pixel_hovertext(geom, image):
    """
    Generate hovertext tags for each pixel
    depending on value.

    Args:
        geom (Geometry): ctapipe cmaera geometry
        image (array): List of values for camera image

    Returns:
        (str array): List of hovertext strings
    """
    pixel_ids = np.arange(len(geom.pix_id)).astype(float)
    pixel_ids_2d = geom.image_to_cartesian_representation(pixel_ids)

    hovertext_2d = np.empty(pixel_ids_2d.shape, dtype=object)
    for i in range(pixel_ids_2d.shape[0]):
        for j in range(pixel_ids_2d.shape[1]):
            val = pixel_ids_2d[i, j]
            if np.isnan(val):
                hovertext_2d[i, j] = ""
            else:
                pid = int(val)
                if image[pid] == 0:
                    hovertext_2d[i, j] = "Dead pixel"
                else:
                    slot, asic, asic_ch = get_pixel_info(pid)
                    hovertext_2d[i, j] = (
                        f"Pixel ID: {pid}<br>"
                        f"Slot: {slot}<br>"
                        f"ASIC: {asic}<br>"
                        f"ASIC Ch: {asic_ch}"
                    )
    return hovertext_2d


def plot_good_pixels(input_file, live_pixels, good_fit_mask, tel_id):
    """Generate a camera map highlighting pixels with good SPE fits.

    Produces an image showing:
      - dead pixels,
      - live pixels with bad fits,
      - live pixels with good fits.

    Includes hover text with pixel metadata.

    Args:
        input_file (str): Path to the data file (used to load geometry).
        live_pixels (array): Indices of camera pixels considered live.
        good_fit_mask (array[bool]): Mask marking good-fit pixels.
        tel_id (int): Telescope ID.

    Returns:
        str: HTML snippet of the Plotly image.
    """

    source, _ = get_source(input_file, 1)
    geom = source.subarray.tel[tel_id].camera.geometry

    image = np.zeros(len(geom.pix_id))
    live_pixels = np.array(live_pixels)
    image[live_pixels[good_fit_mask]] = 2
    image[live_pixels[~good_fit_mask]] = 1

    image_square = geom.image_to_cartesian_representation(image)

    colors = {
        0: [0, 0, 0],  # dead (black)
        1: [50, 50, 50],  # bad (dark grey)
        2: [0, 255, 00],  # good (green)
        "pad": [200, 200, 200],  # padding (grey)
    }

    rgb_image = np.zeros(image_square.shape + (3,), dtype=np.uint8)
    for i in range(image_square.shape[0]):
        for j in range(image_square.shape[1]):
            val = image_square[i, j]
            rgb_image[i, j] = colors["pad"] if np.isnan(val) else colors[int(val)]

    hovertext_2d = get_pixel_hovertext(geom, image)

    fig = go.Figure(go.Image(z=rgb_image, hovertext=hovertext_2d, hoverinfo="text"))
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        width=800,
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return fig.to_html(full_html=False, include_plotlyjs=False)


def plot_param_maps(
    input_file,
    value_lists,
    peak_valley,
    peak_indexes,
    tel_id,
    live_pixels,
    include_pix,
    illum_no,
):
    """Generate camera-map visualisations of SPE parameters with a dropdown selector.

    A camera image is drawn showing live, dead, and excluded pixels. A
    dropdown menu allows selecting any of the extracted SPE parameters, their
    peak-valley ratios, or waveform peak indices.
    Hovertext provides per-pixel metadata.

    Args:
        input_file (str): Filename used to load camera geometry.
        value_lists (dict): Mapping of parameter name to array of per-pixel values.
        peak_valley (array): Peak-to-valley ratios of fitted SPE PDF.
        peak_indexes (array): Waveform peak index positions for each pixel.
        tel_id (int): Telescope ID.
        live_pixels (array): Pixel IDs that are live.
        include_pix (array[bool]): Mask marking included pixels for this plot.
        illum_no (int): Illumination/file index.

    Returns:
        str: HTML snippet containing the Plotly figure.
    """

    source, _ = get_source(input_file, 1)
    geom = source.subarray.tel[tel_id].camera.geometry
    live_pixels = np.array(live_pixels)
    include_pix = np.array(include_pix)

    pixel_mask = np.ones(len(geom.pix_id), dtype=np.uint8)
    pixel_mask[live_pixels] = 2
    excluded_pixels = live_pixels[~include_pix]
    pixel_mask[excluded_pixels] = 3

    pixel_mask_square = geom.image_to_cartesian_representation(pixel_mask)

    colors = {
        0: [200, 200, 200, 255],  # padding (grey)
        1: [0, 0, 0, 255],  # dead (black)
        2: [0, 0, 0, 0],  # live included (transparent)
        3: [255, 255, 255, 255],  # live excluded (white)
    }

    rgba_background = np.zeros(pixel_mask_square.shape + (4,), dtype=np.uint8)
    for i in range(pixel_mask_square.shape[0]):
        for j in range(pixel_mask_square.shape[1]):
            val = pixel_mask_square[i, j]
            if np.isnan(val) or int(val) == 0:
                rgba_background[i, j] = colors[0]
            else:
                rgba_background[i, j] = colors.get(int(val), colors[0])

    pixel_ids = np.arange(len(geom.pix_id)).astype(float)
    pixel_ids_square = geom.image_to_cartesian_representation(pixel_ids)

    hovertext_2d = np.empty(pixel_ids_square.shape, dtype=object)
    for i in range(pixel_ids_square.shape[0]):
        for j in range(pixel_ids_square.shape[1]):
            val = pixel_ids_square[i, j]
            if np.isnan(val):
                hovertext_2d[i, j] = ""
            else:
                pid = int(val)
                mask_val = pixel_mask[pid]
                if mask_val == 1:
                    hovertext_2d[i, j] = "Dead pixel"
                elif mask_val == 3:
                    hovertext_2d[i, j] = "Not included"
                elif mask_val == 2:
                    slot, asic, asic_ch = get_pixel_info(pid)
                    hovertext_2d[i, j] = (
                        f"Pixel ID: {pid}<br>Slot: {slot}<br>ASIC: {asic}<br>ASIC Ch: {asic_ch}"
                    )
                else:
                    hovertext_2d[i, j] = ""

    fig = go.Figure()
    fig.add_trace(go.Image(z=rgba_background))

    buttons = []

    keys = [
        f"eped_{illum_no}",
        "eped_sigma",
        "pe",
        "pe_sigma",
        "opct",
        f"lambda_{illum_no}",
        "reduced_chi2",
        "p_value",
        "peak_valley_ratio",
        "peak_indexes",
    ]

    for i, key in enumerate(keys):
        image = np.full(len(geom.pix_id), np.nan)
        included_live_pixels = live_pixels[include_pix]
        if key == "peak_valley_ratio":
            image[included_live_pixels] = peak_valley[include_pix]
        elif key == "peak_indexes":
            image[included_live_pixels] = peak_indexes[include_pix]
        else:
            image[included_live_pixels] = np.array(value_lists[key])[include_pix]
        image_square = geom.image_to_cartesian_representation(image)

        fig.add_trace(
            go.Heatmap(
                z=image_square,
                colorscale="Viridis",
                colorbar={"title": key},
                hoverinfo="text",
                hovertext=hovertext_2d,
                visible=(i == 0),
                zmin=np.nanmin(image_square),
                zmax=np.nanmax(image_square),
                showscale=True,
            )
        )

        visible = [True] + [j == i for j in range(len(keys))]
        buttons.append(
            {
                "label": key,
                "method": "update",
                "args": [{"visible": visible}, {"coloraxis": None}],
            }
        )

    fig.update_layout(
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.5,
                "y": 1.1,
                "bgcolor": "white",
            }
        ],
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        width=800,
    )
    fig.update_xaxes(showticklabels=False, scaleanchor="y")
    fig.update_yaxes(showticklabels=False)

    return fig.to_html(full_html=False, include_plotlyjs=False)
