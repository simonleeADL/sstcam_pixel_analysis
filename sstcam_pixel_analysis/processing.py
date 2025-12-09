"""
Low-level data extraction and processing tools.

This module has low-level functions used for SPE fitting
and charge-resolution calculation.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ctapipe.calib import CameraCalibrator
from ctapipe.image.extractor import FixedWindowSum
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from spefit.pdf import PDFParameter, SiPMModifiedPoisson
from tqdm import tqdm

from sstcam_pixel_analysis.utilities import (
    baseline_subtract,
    calibrate_extracted_adc,
    get_file_info,
    get_source,
    output_filenames,
)


@dataclass
class FileData:
    """Container for information required to process a single run file.

    Attributes:
        filepath (str): Path to the input run file.
        tel_id (str): Telescope ID associated with the file.
        n_events (int): Number of events in the file.
        run_text (str): HTML-formatted description of the run.
        extracted_adc (np.ndarray): Extracted ADC counts (per-pixel, per-event).
        live_pixels (np.ndarray): IDs for active pixels.
        lambda_guess (float): Initial guess for lambda (expected p.e.).
        pdf (object): SPE PDF used for fitting.
        peak_indexes (np.ndarray): Per-pixel peak sample indices.
    """

    filepath: str
    tel_id: str
    n_events: int
    run_text: str
    extracted_adc: np.ndarray
    live_pixels: np.ndarray
    lambda_guess: float
    pdf: object
    peak_indexes: np.ndarray


def calc_charge_res(extracted, true, mc_poisson=False):
    """Calculate relative charge resolution.

    Args:
        extracted (np.ndarray): Extracted p.e.
        true (np.ndarray): Known true p.e.
        mc_poisson (bool, optional):
            Whether the input source is from simulations
            but should be calculated as Poissonian.

    Returns:
        np.ndarray: Relative charge resolution.
    """

    if mc_poisson:
        return np.sqrt(np.mean((true - extracted) ** 2) + true) / true
    return np.sqrt(np.mean((true - extracted) ** 2)) / true


def process_data_charge_res(dataset, args, pde, lk):
    """Process a dataset for charge-resolution stuff.

    Gets extracted ADC counts, expected p.e. values, and
    mean extracted ADC counts for a given dataset.

    Args:
        dataset (FileData): Dataset to process.
        args (argparse.Namespace): Command-line arguments.
        pde (float): Photon detection efficiency.
        lk (pandas.DataFrame): Lookup table of run settings.

    Returns:
        tuple: '(FileData, expected_pe, mean_extracted_adc)'
    """

    file = Path(dataset.filepath)
    expected_pe = lk["n_ph_meas"][file] * pde

    if expected_pe > 1200 * pde:
        return None

    dataset_result = get_dataset(dataset, args, is_spe=False)
    mean_extracted_adc = np.mean(dataset_result.extracted_adc.T, axis=1)

    return dataset_result, expected_pe, mean_extracted_adc


def get_pe_charge_res(dataset, args, pe_map, expected_pe, transfer_function, lk):
    """Extract calibrated p.e. values and compute charge-resolution.

    Args:
        dataset (FileData): Dataset containing ADC counts.
        args (argparse.Namespace): Command-line arguments.
        pe_map (pandas.DataFrame): Table containing SPE calibration results.
        expected_pe (float): Expected number of p.e.
        transfer_function (dict): Per-pixel calibration lookup table.
        lk (pandas.DataFrame): Lookup table with NSB and expected p.e.

    Returns:
        dict: Extracted p.e., expected p.e., NSB values, per-pixel IDs,
            and summary charge-resolution results.
    """

    extracted_adc = dataset.extracted_adc.T
    extracted_pe = []
    for pix_id, pix_extracted_adc in zip(dataset.live_pixels, extracted_adc):
        calibrated_adc = calibrate_extracted_adc(
            pix_extracted_adc, pix_id, transfer_function, args.window_width
        )
        extracted_pe.append(calibrated_adc)

    extracted_pe = np.array(extracted_pe)

    pixel_ids = dataset.live_pixels
    if args.remove_bad_pixels:
        good_mask = (
            pe_map["good_fit"]
            .reindex(dataset.live_pixels, fill_value=False)
            .to_numpy(bool)
        )
        extracted_pe = extracted_pe[good_mask]
        pixel_ids = pixel_ids[good_mask]

    extracted_pe_list = np.array(extracted_pe).flatten()

    charge_res = calc_charge_res(extracted_pe_list, expected_pe)
    filepath = Path(dataset.filepath)
    nsb = lk.nsb[filepath]

    return {
        "extracted": extracted_pe_list,
        "expected": [expected_pe] * len(extracted_pe_list),
        "nsb": [nsb] * len(extracted_pe_list),
        "pixel_ids": list(np.array(pixel_ids).repeat(extracted_pe.shape[1])),
        "result": (expected_pe, nsb, filepath, charge_res),
    }


def get_dataset(
    dataset, args, lambda_guesses=None, illum_no=None, is_spe=True, show_progress=False
):
    """Extract ADC counts from a file and optionally build an SPE PDF.

    Reads from existing checkpoint files when available, otherwise extracts
    ADC counts from R1 waveforms. Estimates or applies lambda (expected p.e.)
    guesses and constructs an SPE PDF if requested.

    Args:
        dataset (FileData): Dataset object to populate.
        args (argparse.Namespace): Command-line arguments.
        lambda_guesses (list[float], optional): Initial lambda guesses for
            all illuminations. Defaults to None.
        illum_no (int, optional): Illumination index for PDF construction.
        is_spe (bool, optional): Whether to construct an SPE PDF. Defaults to True.
        show_progress (bool, optional): Show a progress bar. Defaults to False.

    Returns:
        FileData: Updated dataset with extracted ADC and PDF.
    """

    f = dataset.filepath
    n_events = dataset.n_events
    max_events = args.max_events

    if max_events is None:
        max_events = n_events
    else:
        max_events = min(n_events, max_events)

    need_to_read = True

    if is_spe:
        report = "SPE"
    else:
        report = "charge_res"

    all_extracted_adc_file, _, _ = output_filenames(f, args.output_dir, report=report)

    if os.path.isfile(all_extracted_adc_file) and not args.overwrite:
        data = np.load(all_extracted_adc_file)
        all_extracted_adc = data["all_extracted_adc"]
        dataset.live_pixels = data["live_pixels"]
        dataset.peak_indexes = data["peak_indexes"]
        need_to_read = False

    if need_to_read:
        all_extracted_adc = read_data(
            dataset,
            args,
            max_events=max_events,
            show_progress=show_progress,
            extractor_params={
                "window_width": args.window_width,
                "window_shift": int(args.window_width / 2),
            },
        )
        os.makedirs(os.path.dirname(all_extracted_adc_file), exist_ok=True)
        np.savez(
            all_extracted_adc_file,
            all_extracted_adc=all_extracted_adc,
            live_pixels=dataset.live_pixels,
            peak_indexes=dataset.peak_indexes,
        )

    show_plot = getattr(args, "peak_helper", False)
    peak_location = peak_helper(all_extracted_adc, show_plot=show_plot)
    lambda_guess = 1

    if is_spe:
        if lambda_guesses is None:
            lambda_guess = peak_location / args.pe_guess
            if lambda_guess < 0:
                lambda_guess = 1
            good_guess = False
        else:
            lambda_guess = lambda_guesses[illum_no]
            good_guess = True
        pdf = sipm_pdf(
            pe_guess=args.pe_guess,
            lambda_guess=lambda_guess,
            illum_no=illum_no,
            good_guess=good_guess,
        )
    else:
        pdf = None

    dataset.extracted_adc = all_extracted_adc.T
    dataset.lambda_guess = lambda_guess
    dataset.pdf = pdf

    return dataset


def fixed_window_extractor(source, p, correction=False):
    """Returns a ctapipe FixedWindowSum extractor.

    Args:
        source (EventSource): Event source object.
        p (dict): Extraction parameters containing peak index, window width,
            and window shift.
        correction (bool, optional): Apply integration correction. Defaults to False.

    Returns:
        FixedWindowSum: ctapipe extractor.
    """

    ext = FixedWindowSum(
        subarray=source.subarray,
        peak_index=p["peak_index"],
        window_width=p["window_width"],
        window_shift=p["window_shift"],
        apply_integration_correction=correction,
    )
    return ext


def get_peaks_and_offs(source, tel_id, max_events, show_progress=False):
    """Obtain waveform peak indexes for each pixel.

    Iterates over events, computes the average waveform, finds peak indices,
    and detects 'off' pixels.

    Args:
        source (EventSource): Input event source.
        tel_id (int): Telescope ID.
        max_events (int): Maximum events to inspect.
        show_progress (bool): Whether to display a progress bar.

    Returns:
        tuple: (peak_indexes, off_pixels)
    """

    n_events = 0
    if show_progress:
        iterator = tqdm(source, total=max_events, desc="Reading waveforms", leave=False)
    else:
        iterator = source

    for event in iterator:
        if event.meta.get("stale") or event.meta.get("missing_packets"):
            continue
        if n_events == max_events:
            break
        n_events += 1
        baseline_subtracted_event = baseline_subtract(event)
        if n_events == 1:
            average_waveform = baseline_subtracted_event.r1.tel[tel_id].waveform[0]
        else:
            average_waveform += baseline_subtracted_event.r1.tel[tel_id].waveform[0]

    peak_indexes = np.argmax(average_waveform, axis=1)

    average_waveform = average_waveform / n_events
    global_max = np.max(average_waveform)
    row_max = np.max(average_waveform[:, 20:50], axis=1)

    off_pixels = np.where(row_max < (global_max / 3))[0]

    return peak_indexes, off_pixels


def sipm_pdf(pe_guess=15.0, lambda_guess=2.0, illum_no=0, good_guess=False):
    """
    Initialised SiPM probability density function
    for the fitter to use for fitting the SPE.

    Includes guessed starting values and ranges for
    eped, eped_sigma, pe, pe_sigma, lambda, opct

    Args:
        pe_guess (float): Guess for the extracted ADC of 1 p.e.. Defaults to 15.
        lambda_guess (float): Guess for the expected photon brightness. Defaults to 2.
        illum_no (int): Illumination/file number. Defaults to 0.
        good_guess (bool): Whether the lambda guess is reliable.

    Returns:
        SiPMModifiedPoisson: SiPM PDF object
    """
    if good_guess:
        lambda_lim = (lambda_guess * 0.7, lambda_guess * 1.3)
        pe_lim = (pe_guess * 0.7, pe_guess * 1.3)
    else:
        lambda_lim = (0.1, lambda_guess * 5)
        pe_lim = (pe_guess * 0.2, pe_guess * 4)

    eped_init = 0.0
    eped_lim = (-(pe_guess * 3), pe_guess * 3)

    eped_sig_init = pe_guess / 4
    eped_sig_lim = (pe_guess / 50, pe_guess * 0.75)

    pe_sig_init = pe_guess / 4
    pe_sig_lim = (pe_guess / 50, pe_guess * 0.75)

    pdf = SiPMModifiedPoisson(
        eped=PDFParameter(
            name=f"eped_{illum_no}", initial=eped_init, limits=eped_lim, fixed=False
        ),
        eped_sigma=PDFParameter(
            name="eped_sigma",
            initial=eped_sig_init,
            limits=eped_sig_lim,
            fixed=False,
        ),
        pe=PDFParameter(
            name="pe",
            initial=pe_guess,
            limits=pe_lim,
            fixed=False,
        ),
        pe_sigma=PDFParameter(
            name="pe_sigma", initial=pe_sig_init, limits=pe_sig_lim, fixed=False
        ),
        lambda_=PDFParameter(
            name=f"lambda_{illum_no}",
            initial=lambda_guess,
            limits=lambda_lim,
            fixed=False,
        ),
        opct=PDFParameter(name="opct", initial=0.1, limits=(0.0, 0.3), fixed=False),
    )
    return pdf


def peak_helper(all_extracted_adc, show_plot=False):
    """
    A tool to guess the location of highest peak.
    Alternatively, to show a histoagram of
    all extracted ADC so guessing the pe is easier.

    Args:
        all_extracted_adc (array): Array of extracted ADC
        show_plot (bool, optional): Shows a histogram of extracted ADC. Defaults to False.

    Returns:
        (float): Extracted ADC location of highest peak
    """
    all_extracted_adc = all_extracted_adc.ravel()
    all_extracted_adc = np.sort(all_extracted_adc)

    range_min = np.percentile(all_extracted_adc, 1)
    range_max = np.percentile(all_extracted_adc, 99)
    hist, bin_edges = np.histogram(
        all_extracted_adc, bins=100, range=(range_min, range_max), density=True
    )

    if show_plot:
        plt.figure(figsize=(5, 3))
        plt.step(bin_edges[:-1], hist, label="Data", where="pre", color="grey")
        plt.xlabel("mV*10ns")
        plt.ylabel("Normalised counts")
        plt.grid()
        plt.show()

    peak_location = bin_edges[np.argmax(hist)]
    return peak_location


def initialise_data(filename, args):
    """
    Initialise a FileData object, including the peak sample list
    and the baseline correction of requested.

    Args:
        filename (str): Filename of lab run
        args (argparse.Namespace): Command-line arguments

    Returns:
        FileData: Initialised FileData object
    """
    tel_id, run_text, n_events = get_file_info(filename)

    max_events = args.max_events

    if max_events is None:
        max_events = n_events
    else:
        max_events = min(n_events, max_events)

    source, _ = get_source(filename, max_events)
    for event in source:
        dead_pixels_mask = event.r1.tel[tel_id].pixel_status == False
        break
    source, _ = get_source(filename, max_events)

    peak_indexes_all, off_pixels = get_peaks_and_offs(source, tel_id, max_events)

    n_pixels = len(dead_pixels_mask)
    n_existing_pixels = n_pixels - sum(dead_pixels_mask)
    n_on_pixels = n_pixels - len(off_pixels)
    n_off_pixels = n_existing_pixels - n_on_pixels

    if n_off_pixels > 0:
        if n_on_pixels / n_off_pixels < 0.2:
            dead_pixels_mask[off_pixels] = True
    live_pixels = np.where(~dead_pixels_mask)[0].tolist()

    peak_indexes = peak_indexes_all[~dead_pixels_mask]

    dataset = FileData(
        filename,
        tel_id,
        n_events,
        run_text,
        None,
        live_pixels,
        None,
        None,
        peak_indexes,
    )

    return dataset


def read_data(
    dataset,
    args,
    max_events=None,
    extractor_params=None,
    show_progress=False,
):
    """
    Perform ADC count extraction for a given file.

    Args:
        dataset (FileData):
            Initialised FileData object.
        args (argparse.Namespace):
            Command line arguments
            (used for leave_time_skew and leave_baseline)
        max_events (int, optional):
            Max number of events to process. Defaults to None.
        extractor_params (dict, optional):
            Fixed Window extraction parameters.
            Defaults to {"window_width": 16, "window_shift": 8}.
        show_progress (bool, optional):
            Show progress bar. Defaults to False.

    Returns:
        all_extracted_adc: Array of extracted ADC counts for all pixels and events
    """

    if extractor_params is None:
        extractor_params = {"window_width": 16, "window_shift": 8}

    tel_id = dataset.tel_id

    filename = dataset.filepath
    source, correction = get_source(filename, max_events)

    peak_index = int(np.median(dataset.peak_indexes))
    extractor_params["peak_index"] = peak_index
    peak_shifts = dataset.peak_indexes - peak_index

    image_extractor = fixed_window_extractor(
        source, extractor_params, correction=correction
    )

    calib = CameraCalibrator(subarray=source.subarray, image_extractor=image_extractor)

    if show_progress:
        iterator = tqdm(
            source, total=max_events, desc="Extract ADC", leave=False, flush=True
        )
    else:
        iterator = source

    adc_list = []
    for event in iterator:
        if event.meta.get("stale") or event.meta.get("missing_packets"):
            continue
        if not args.leave_time_skew:
            waveforms = event.r1.tel[tel_id].waveform[0]
            new_waveforms = waveforms.copy()
            for i, pix_no in enumerate(dataset.live_pixels):
                new_waveforms[pix_no] = np.roll(waveforms[pix_no], -peak_shifts[i])
            event.r1.tel[tel_id].waveform[0] = new_waveforms
        if not args.leave_baseline:
            event = baseline_subtract(event)
        calib(event)
        adc_list.append(event.dl1.tel[tel_id].image)
    all_extracted_adc = np.vstack(adc_list).T
    all_extracted_adc = all_extracted_adc[dataset.live_pixels]

    return all_extracted_adc


def get_bad_fit_mask(value_lists):
    """
    Generates a mask for pixels with a bad fit.
    bad_1: pe_sigma > 0.5 * pe
    bad_2: abs(pe - pe_med) > 3 * np.std(pe)
    bad_3: abs(pe_sigma - pe_s_med) > 3 * np.std(pe_sigma)

    Args:
        value_lists (dict): Dict of arrays of extracted parameters

    Returns:
        np.ndarray: Mask of pixels with bad fits
    """
    pe = np.array(value_lists["pe"])
    pe_sigma = np.array(value_lists["pe_sigma"])
    reduced_chi2 = np.array(value_lists["reduced_chi2"])

    pe_med = np.median(pe)
    pe_s_med = np.median(pe_sigma)

    bad_1 = pe_sigma > 1 * pe
    bad_2 = abs(pe - pe_med) > 3 * np.std(pe)
    bad_3 = abs(pe_sigma - pe_s_med) > 6 * np.std(pe_sigma)
    bad_4 = reduced_chi2 > 50

    return bad_1 | bad_2 | bad_3 | bad_4


def get_peak_valley_ratios(value_lists, fitter):
    """Compute peak-to-valley ratios for all pixels and illuminations.

    For each pixel, finds adjacent peaks in the fitted PDF and computes
    a peak-to-valley metric, which is used to assess SPE fit quality.

    Args:
        value_lists (dict): Dictionary of per-pixel fit parameters.
        fitter: SPE fitter object containing per-pixel fits.

    Returns:
        np.ndarray: Array shaped (n_illuminations * n_pixels) with ratios.
    """
    peak_valley_ratios = []

    for ipix, _ in enumerate(fitter.pixel_arrays):
        pixfit = fitter.pixel_arrays[ipix]
        tmp_peak_valley_ratios = []

        for illum_no, _ in enumerate(pixfit):
            illum = pixfit[illum_no]
            fit_x = illum["fit_x"]
            fit_y = illum["fit_y"]

            eped = value_lists[f"eped_{illum_no}"][ipix]
            pe = value_lists["pe"][ipix]

            peaks, _ = find_peaks(fit_y, prominence=0.000001, distance=10)

            max_peak_valley_ratio = 1.0

            if len(peaks) >= 2:
                best_pair_sum = -1.0
                best_pe_a_height = 0
                best_pe_b_height = 0
                dip_search_start_index = 0
                dip_search_end_index = 0

                for i in range(len(peaks) - 1):
                    pe_a_index = peaks[i]
                    pe_b_index = peaks[i + 1]

                    pe_a_height = fit_y[pe_a_index]
                    pe_b_height = fit_y[pe_b_index]

                    current_sum = pe_a_height + pe_b_height

                    if current_sum > best_pair_sum:
                        best_pair_sum = current_sum
                        best_pe_a_height = pe_a_height
                        best_pe_b_height = pe_b_height
                        dip_search_start_index = pe_a_index
                        dip_search_end_index = pe_b_index

                search_region = fit_y[dip_search_start_index:dip_search_end_index]
                min_relative_index = np.argmin(search_region)
                dip_absolute_index = dip_search_start_index + min_relative_index
                dip_height = fit_y[dip_absolute_index]

                if dip_height == 0:
                    max_peak_valley_ratio = 0.0
                else:
                    max_peak_valley_ratio = (
                        (best_pe_a_height + best_pe_b_height) / 2
                    ) / dip_height

            else:

                pe_1_value = eped + 1 * pe
                pe_2_value = eped + 2 * pe
                dip_value = eped + pe * (1 + 0.5)

                x_1 = np.argmin(np.abs(fit_x - pe_1_value))
                pe_1_height = fit_y[x_1]

                x_2 = np.argmin(np.abs(fit_x - pe_2_value))
                pe_2_height = fit_y[x_2]

                x_dip = np.argmin(np.abs(fit_x - dip_value))
                dip_height = fit_y[x_dip]

                if dip_height == 0:
                    max_peak_valley_ratio = 0.0
                else:
                    max_peak_valley_ratio = (
                        (pe_1_height + pe_2_height) / 2
                    ) / dip_height

            tmp_peak_valley_ratios.append(max_peak_valley_ratio)

        peak_valley_ratios.append(tmp_peak_valley_ratios)

    return np.array(peak_valley_ratios).T
