import json
import os
import threading
import time
from dataclasses import dataclass
from glob import glob
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

from .plotting import (
    get_param_text,
    plot_all_fits_plotly,
    plot_charge_res,
    plot_charge_res_relative,
    plot_dispersion,
    plot_fit_plotly,
    plot_good_pixels,
    plot_param_maps,
    plot_value_lists_plotly,
)
from .processing import (
    FileData,
    get_bad_fit_mask,
    get_dataset,
    get_peak_valley_ratios,
    initialise_data,
    get_pe_charge_res,
    process_data_charge_res,
)
from .utilities import (
    get_file_info,
    get_processing_text,
    get_run_text_chres,
    get_value_lists,
    load_calibration,
    output_filenames,
    spin,
    write_pixel_spe_table,
    get_lookup_charge_res,
    too_bright,
)

# Reading in configs from config.yaml

SCRIPT_DIR = Path(__file__).resolve().parent

with open(f"{SCRIPT_DIR}/config.yaml", "r") as _f:
    _cfg = yaml.safe_load(_f)

HIST_BINS = _cfg["hist_bins"]
GOOD_PEAK_RATIO = _cfg["good_peak_ratio"]
PDE = _cfg["PDE"]


def initialise_datasets(args, linear=False):
    """
    Initialise the datasets of the runs.

    Args:
        args (args): Command-line arguments
        linear (bool, optional): Process files non-parallel. Defaults to False.

    Returns:
        list: List of FileData objects
    """
    if hasattr(args, "input_file"):
        input_files = args.input_file
    else:
        base_dir = args.input_dir
        input_files = glob(f"{base_dir}*r1.tio")
        input_files.sort()
    
    if hasattr(args, "input_dir"):
        lk = get_lookup_charge_res(args)
        for f in input_files.copy():
            if too_bright(lk,f):
                input_files.remove(f)

    need_to_initialise = False
    for f in input_files:
        if args.overwrite:
            need_to_initialise = True
            break
        all_extracted_adc_file, _, _ = output_filenames(f, args.output_dir)
        if not os.path.isfile(all_extracted_adc_file):
            need_to_initialise = True

    num_files = len(input_files)

    if need_to_initialise:
        if linear:
            datasets = []
            for f in tqdm(input_files):
                d = initialise_data(f, args)
                datasets.append(d)
            return datasets
        else:
            done = False
            start_time = time.time()
            t = threading.Thread(
                target=spin, args=(lambda: done, start_time, num_files)
            )
            t.start()

            try:
                parallel = Parallel(n_jobs=-1, backend="loky")
                tasks = (delayed(initialise_data)(f, args) for f in input_files)
                return list(parallel(tasks))
            finally:
                done = True
                t.join()
    else:
        datasets = []
        for f in input_files:
            tel_id, run_text, n_events = get_file_info(f)
            dataset = FileData(
                f, tel_id, n_events, run_text, None, None, None, None, None, None
            )
            datasets.append(dataset)
        return datasets


def get_datasets_spe(datasets, args, lambda_guesses, linear=False):
    """
    Read in datasets for SPE fitting in parallel,
    whether from checkpoints or processing from scratch.
    It also shows a spinning animation because
    a progress bar is infeasible.

    Args:
        args: Command-line arguments for SPE fitting

    Returns:
        list: A list of datasets in FileData class (defined in plotting.py)
    """
    num_files = len(args.input_file)

    linear=True

    if lambda_guesses is not None:
        for dataset in datasets:
            dataset.peak_indexes = datasets[np.argmax(lambda_guesses)].peak_indexes

    if linear:
        new_datasets = []
        for i, dataset in tqdm(enumerate(datasets), total=len(datasets)):
            d = get_dataset(dataset, args, illum_no=i, lambda_guesses=lambda_guesses)
            new_datasets.append(d)
        return new_datasets
    else:
        done = False
        start_time = time.time()
        t = threading.Thread(target=spin, args=(lambda: done, start_time, num_files))
        t.start()

        try:
            parallel = Parallel(n_jobs=-1, backend="loky")
            tasks = (
                delayed(get_dataset)(
                    dataset, args, illum_no=i, lambda_guesses=lambda_guesses
                )
                for i, dataset in enumerate(datasets)
            )
            return list(parallel(tasks))
        finally:
            done = True
            t.join()


def do_fitting_spe(args, datasets):
    """
    Perform the SPE fit.

    Args:
        args: Command-line arguments for SPE fitting
        datasets: A list of datasets in FileData class (defined in plotting.py)

    Returns:
        FitResults: The results of the fit in FitResults class (defined here)
    """
    from spefit.fitter import CameraFitter
    from spefit.pdf import PDFSimultaneous

    @dataclass
    class FitResults:
        fitter: CameraFitter
        value_lists: list
        peak_valley: np.ndarray
        good_fit_masks: np.ndarray
        hist_range: range

    if len(datasets) == 1:
        final_pdf = datasets[0].pdf
    else:
        final_pdf = PDFSimultaneous([f.pdf for f in datasets])

    max_lambda_guess = max(f.lambda_guess for f in datasets)

    hist_range = (
        -(args.pe_guess * 1.5),
        args.pe_guess * (max_lambda_guess + 4 * np.sqrt(max_lambda_guess)),
    )

    fitter = CameraFitter(
        pdf=final_pdf, n_bins=HIST_BINS, range_=hist_range, cost_name="BinnedNLL"
    )

    all_extracted_adc_datasets = [f.extracted_adc for f in datasets]
    fitter.multiprocess(all_extracted_adc_datasets, n_processes=3)

    n_files = len(datasets)
    value_lists = get_value_lists(fitter, n_files)
    peak_valley_ratios = get_peak_valley_ratios(value_lists, fitter)
    bad_fit_masks = np.array([get_bad_fit_mask(value_lists)] * n_files)
    good_fit_masks = (peak_valley_ratios >= GOOD_PEAK_RATIO) & (~bad_fit_masks)

    return FitResults(
        fitter, value_lists, peak_valley_ratios, good_fit_masks, hist_range
    )


def write_reports_spe(args, datasets, fit):
    """
    Generate the HTML reports for the SPE fit.

    Three reports are generated for each file:

    ALL: All pixels
    GOOD: Pixels with a "good" SPE fit
    BAD: Pixels without a "good" SPE fit

    The goodness of the fit is defined by whether the
    average height between the 1st and 2nd p.e. peaks is
    at least 8 percent higher than the trough inbetween.
    This is configurable in config.yaml.

    Args:
        args: Command-line arguments for SPE fitting
        datasets: A list of datasets in FileData class (defined in plotting.py)
        fit: The results of the fit in FitResults class (defined in do_fitting_spe)
    """
    from jinja2 import Template

    for dataset, good_fit_mask in zip(datasets, fit.good_fit_masks):
        dataset.run_text += (
            f"<b>Good/Bad pixels</b>: {sum(good_fit_mask)}/{sum(~good_fit_mask)}"
        )

    _, _, output_dir = output_filenames(datasets[0].filepath, args.output_dir)
    csv_output = f"{output_dir}/spe_output.csv"

    live_pixels = datasets[0].live_pixels

    write_pixel_spe_table(fit, live_pixels, csv_output)

    suffixes = ["ALL", "GOOD", "BAD"]
    loop_items = list(product(enumerate(args.input_file), suffixes))

    processing_text = get_processing_text(args)
    processing_text += f"<b>Fitted pixels</b>: {len(live_pixels)}"

    tel_id = datasets[0].tel_id

    for (illum_no, f), suffix in tqdm(
        loop_items, total=len(datasets) * len(suffixes), desc="Saving reports"
    ):
        _, report_fl, _ = output_filenames(f, args.output_dir)

        report_filename = report_fl
        if suffix == "ALL":
            include_pix = np.array([True] * len(live_pixels))
        else:
            report_filename = report_fl.replace(".html", f"_{suffix}.html")
            if suffix == "GOOD":
                include_pix = fit.good_fit_masks[illum_no]
            if suffix == "BAD":
                include_pix = ~fit.good_fit_masks[illum_no]

        if os.path.exists(report_filename):
            os.remove(report_filename)
        if not include_pix.any():
            continue

        selected_ipix = np.where(include_pix)[0]

        pixels = []

        for ipix in tqdm(selected_ipix, desc="Plotting fits", leave=False):
            pix_no = live_pixels[ipix]

            param_text = get_param_text(ipix, illum_no, include_pix, fit)

            c = datasets[illum_no].extracted_adc.T[ipix]
            pixel_plot, pixel_text = plot_fit_plotly(
                c, fit, param_text, ipix, pix_no, illum_no, HIST_BINS
            )
            pixels.append([pixel_plot, pixel_text])

        jinja_data = {}
        jinja_data["param_dist"] = plot_value_lists_plotly(
            fit.value_lists, illum_no, include_pix
        )
        jinja_data["good_pixels"] = plot_good_pixels(
            args.input_file[0],
            datasets[illum_no].live_pixels,
            fit.good_fit_masks[illum_no],
            tel_id,
        )

        jinja_data["value_map"] = plot_param_maps(
            args.input_file[0],
            fit.value_lists,
            fit.peak_valley[illum_no],
            datasets[illum_no].peak_indexes,
            tel_id,
            datasets[illum_no].live_pixels,
            include_pix,
            illum_no,
        )
        jinja_data["fits_overview"] = plot_all_fits_plotly(
            datasets[illum_no].live_pixels, fit.fitter, illum_no, include_pix
        )

        jinja_data["pixels_type"] = suffix
        jinja_data["processing_text"] = processing_text
        jinja_data["run_text"] = datasets[illum_no].run_text
        jinja_data["pixels"] = pixels

        report_template = f"{SCRIPT_DIR}/../templates/spe_report_template.html"
        os.makedirs(os.path.dirname(report_filename), exist_ok=True)

        with open(report_filename, "w", encoding="utf-8") as f:
            with open(report_template) as template_file:
                spe_report_template = Template(template_file.read())
                f.write(spe_report_template.render(jinja_data))

def get_charge_res_output(datasets, args):
    """
    Extract the images of events from dynamic range runs
    and calculate charge resolution.

    Args:
        args: Command-line arguments for charge resolution

    Returns:
        df (pandas df): Pandas dataframe of extracted charge resolutions
        df_2d (pandas df): Pandas dataframe of all extracted ADC values
        run_text (str): Text description of this run
    """

    run_text = get_run_text_chres(datasets[0].filepath)
    lk = get_lookup_charge_res(args)

    ref_spe = pd.read_csv(args.ref_spe)
    pe_map = ref_spe.set_index("pixel_no")[["pe", "good_fit"]]

    charge_res_results = []
    all_extracted = []
    all_expected = []
    all_pixel_ids = []
    all_nsb = []
    nsb_zero = []

    for dataset in datasets.copy():
        filepath = Path(dataset.filepath)
        n_ph = lk["n_ph_meas"][filepath]
        nsb = lk["nsb"][filepath]
        if n_ph > 1200:
            datasets.remove(dataset)
        else:
            if nsb == 0.0:
                nsb_zero.append(True)
            else:
                nsb_zero.append(False)
    
    nsb_zero = np.array(nsb_zero)

    parallel_results = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_data_charge_res)(dataset, args, PDE, lk)
        for dataset in tqdm(datasets, desc="Processing files")
    )

    valid_results = [res for res in parallel_results if res is not None]

    datasets = [res[0] for res in valid_results]
    expected_pe_list = [res[1] for res in valid_results]

    all_mean_extracted_adc = np.array([res[2] for res in valid_results])
    all_mean_extracted_adc = all_mean_extracted_adc[nsb_zero].T
    expected_pe_list = np.array(expected_pe_list)

    tf_filepath = args.output_dir + f"/transfer_functions.json"

    if not os.path.exists(tf_filepath) or args.overwrite:
        cal_data = {}
        for pix_id, pix_mean_extracted_adc in zip(
            datasets[0].live_pixels, all_mean_extracted_adc
        ):
            y_tf = pix_mean_extracted_adc / expected_pe_list[nsb_zero]
            x_tf = pix_mean_extracted_adc
            smooth = np.convolve(y_tf, [0.5, 1, 0.5], mode="same") / np.convolve(
                np.ones_like(y_tf), [0.5, 1, 0.5], mode="same"
            )
            cal_data.setdefault(int(args.window_width), {})[int(pix_id)] = {
                "extracted_adc": x_tf.tolist(),
                "adc_per_pe": smooth.tolist(),
            }

        with open(tf_filepath, "w") as f:
            json.dump(cal_data, f)

    transfer_function = load_calibration(tf_filepath)

    parallel = Parallel(n_jobs=-1, backend="loky")
    output = parallel(
        delayed(get_pe_charge_res)(
            dataset, args, pe_map, expected_pe, transfer_function, lk
        )
        for dataset, expected_pe in tqdm(
            zip(datasets, expected_pe_list), total=len(datasets), desc="Reading runs"
        )
    )

    for out in output:
        charge_res_results.append(out["result"])
        all_extracted.extend(out["extracted"])
        all_expected.extend(out["expected"])
        all_pixel_ids.extend(out['pixel_ids'])
        all_nsb.extend(out["nsb"])

    df = pd.DataFrame.from_records(
        charge_res_results, columns=["expected_pe", "nsb", "filename", "charge_res"]
    )

    df_2d = pd.DataFrame(
        np.column_stack((all_pixel_ids, all_extracted, all_expected, all_nsb)),
        columns=["pixel_id","extracted_pe", "expected_pe", "nsb"],
    )

    return df, df_2d, run_text


def write_report_charge_res(args, df, df_2d, run_text):
    """
    Write an HTML report of the extracted charge resolution,
    which is split by induced NSB.
    Also includes a 2D histogram of expected vs relative extracted p.e.

    Args:
        args: Command-line arguments for charge resolution
        df (pandas df): Pandas dataframe of extracted charge resolutions
        df_2d (pandas df): Pandas dataframe of all extracted charges
        run_text (str): Text description of this run
    """
    from jinja2 import Template

    pixels_type = "GOOD" if args.remove_bad_pixels else "ALL"
    report_filename = args.output_dir + f"/CHARGE_RES_{pixels_type}.html"

    jinja_data = {}

    jinja_data["pixels_type"] = pixels_type
    jinja_data["run_text"] = run_text
    jinja_data["processing_text"] = get_processing_text(args, mode="res")

    jinja_data["charge_res_plot"] = plot_charge_res(df)
    jinja_data["charge_res_plot_scaled"] = plot_charge_res_relative(df_2d)
    jinja_data["nsb_hists"] = plot_dispersion(df_2d)

    report_template = f"{SCRIPT_DIR}/../templates/charge_res_report_template.html"
    os.makedirs(os.path.dirname(report_filename), exist_ok=True)

    with open(report_filename, "w", encoding="utf-8") as f:
        with open(report_template) as template_file:
            spe_report_template = Template(template_file.read())
            f.write(spe_report_template.render(jinja_data))
