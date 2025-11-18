import os
from dataclasses import dataclass

import numpy as np
from ctapipe.calib import CameraCalibrator
from ctapipe.image.extractor import FixedWindowSum
from ctapipe_io_sstcam import SSTCAMEventSource
from matplotlib import pyplot as plt
from spefit.pdf import PDFParameter, SiPMModifiedPoisson
from tqdm import tqdm
from pathlib import Path
from .utilities import get_file_info, output_filenames


@dataclass
class FileData:
    filepath: str
    tel_id: str
    n_events: int
    run_text: str
    charges: np.ndarray
    live_pixels: np.ndarray
    lambda_guess: float
    pdf: object
    peak_indexes: np.ndarray
    baseline_corrections: np.ndarray


def calc_charge_res(extracted, true, poisson=False):
    """
    Calculate relative charge resolution.

    Args:
        extracted (numpy array): Extracted charges
        true (numpy array): Known true charges
        poisson (bool, optional): Whether the input source is poissonian (True for lab data). Defaults to False.

    Returns:
        numpy array: Relative charge resolution.
    """
    if poisson:
        return np.sqrt(np.mean((true - extracted) ** 2) + true) / true
    return np.sqrt(np.mean((true - extracted) ** 2)) / true


def process_file_charge(dataset, args, pe_map, lk):
    """
    Extract the charges from a given R1 file
    and their associated expected values and NSB.

    Args:
        filename (str): Name of input file
        args: Command-line arguments
        pe_map (pandas df): Refernce table for extracted PE calibration
        lk (lookup): Reference table for NSB and expected PE

    Returns:
        dict: Extracted charges, expected charges, nsb 
    """
    filename = Path(dataset.filepath)
    dataset = get_dataset(dataset, args, do_pdf=False)
    pe_values = pe_map.loc[dataset.live_pixels, "pe"].to_numpy()
    calibrated_pe = dataset.charges.T / pe_values[:, None]

    if args.remove_bad_pixels:
        good_mask = pe_map.loc[dataset.live_pixels, "good_fit"].to_numpy(dtype=bool)
        calibrated_pe = calibrated_pe[good_mask]

    n_ph = lk.n_ph_meas[filename]
    expected_pe = n_ph / 2
    extracted_pe_list = calibrated_pe.flatten()

    charge_res = calc_charge_res(extracted_pe_list, expected_pe, poisson=True)

    return {
        "extracted": extracted_pe_list,
        "expected": [expected_pe] * len(extracted_pe_list),
        "nsb": [lk.nsb[filename]] * len(extracted_pe_list),
        "result": (expected_pe, lk.nsb[filename], filename, charge_res),
    }


def get_dataset(dataset, args, lambda_guesses=None, illum_no=None, do_pdf=True, show_progress=False):
    """
    Get the extracted charges of a given file.

    Args:
        dataset (FileData): Initialised FileData object
        args: Command-line arguments
        lambda_guesses (list, optional): List of . Defaults to None.
        illum_no (int, optional): Illumination number. Defaults to None.
        do_pdf (bool, optional): Whether to add an SPE PDF. Defaults to True.
        show_progress (bool, optional): Show a progress bar. Defaults to False.

    Returns:
        FileData: The filled-out FileData object with extracted charges
    """
    f = dataset.filepath
    n_events = dataset.n_events
    max_events = args.max_events

    if max_events is None:
        max_events = n_events
    else:
        max_events = min(n_events, max_events)

    need_to_read = True

    all_charges_file, _, _ = output_filenames(f, args.output_dir)

    if os.path.isfile(all_charges_file) and not args.overwrite:
        data = np.load(all_charges_file)
        all_charges_lab = data["all_charges_lab"]
        dataset.live_pixels = data["live_pixels"]
        dataset.peak_indexes = data["peak_indexes"]
        need_to_read = False

    if need_to_read:
        all_charges_lab = read_lab_data(
            dataset,
            args,
            max_events=max_events,
            show_progress=show_progress,
            extractor_params={
                "window_width": args.window_width,
                "window_shift": int(args.window_width / 2),
            },
        )
        os.makedirs(os.path.dirname(all_charges_file), exist_ok=True)
        np.savez(
            all_charges_file,
            all_charges_lab=all_charges_lab,
            live_pixels=dataset.live_pixels,
            peak_indexes=dataset.peak_indexes,
        )

    show_plot = getattr(args, "peak_helper", False)
    peak_location = peak_helper(all_charges_lab, show_plot=show_plot)
    lambda_guess = 1

    if do_pdf:
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

    dataset.charges = all_charges_lab.T
    dataset.lambda_guess = lambda_guess
    dataset.pdf = pdf

    return dataset


def fixed_window_extractor(source, p, correction=False):
    """
    Returns a ctapipe Fixed Window Extractor
    for a given list of extraction parameters

    Args:
        source (EventSource):
        p (dict): Extraction parameters

    Returns:
        (FixedWindowSum) : ctapipe extractor
    """
    ext = FixedWindowSum(
        subarray=source.subarray,
        peak_index=p["peak_index"],
        window_width=p["window_width"],
        window_shift=p["window_shift"],
        apply_integration_correction=correction,
    )
    return ext


def get_base_peak(source, tel_id, max_events, show_progress=False):
    """
    Returns a rudimentary baseline correction for each pixel
    (based off of the average of the first 5 samples of the
    average waveform across all events), the waveform peak
    indexes, and which pixels are turned off (based of if
    the peak waveform average is less than half the global
    max, in the first 50 samples)

    Args:
        source (EventSource): ctapipe EventSource
        tel_id (int): Telescope ID
        max_events (int): Maximum number of events to process
        show_progress (bool): Whether to show progress bar

    Returns:
        baseline_correction (int array): Baseline correction factor
        peak_indexes (int array): Peak indexes
        off_pixels (bool array): Mask of pixels that are off
    """
    n = 0
    if show_progress:
        iterator = tqdm(source, total=max_events, desc="Reading waveforms", leave=False)
    else:
        iterator = source

    for event in iterator:
        if n == max_events:
            break
        n += 1
        if n == 1:
            arr_base = event.r1.tel[tel_id].waveform[0]
        else:
            arr_base += event.r1.tel[tel_id].waveform[0]
    avg_noise = np.mean(arr_base[:, :5], axis=1)
    baseline_correction = -(avg_noise / max_events)
    peak_indexes = np.argmax(arr_base, axis=1)

    calib_average = (arr_base / max_events) + baseline_correction[:, np.newaxis]
    global_max = np.max(calib_average)
    row_max = np.max(calib_average[:, :50], axis=1)
    off_pixels = np.where(row_max < (global_max / 2))[0]

    return baseline_correction, peak_indexes, off_pixels


def sipm_pdf(pe_guess=15, lambda_guess=2, illum_no=0, good_guess=False):
    """
    Initialised SiPM probability density function
    for the fitter to use for fitting the SPE.

    Includes guessed starting values and ranges for
    eped, eped_sigma, pe, pe_sigma, lambda, opct

    Args:
        pe_guess (int): Guess for the extracted charge of 1 p.e.. Defaults to 15.
        lambda_guess (int): Guess for the expected photon brightness. Defaults to 2.
        illum_no (int): Illumination/file number. Defaults to 0.

    Returns:
        SiPMModifiedPoisson: SiPM PDF
    """
    if good_guess:
        lambda_lim = (lambda_guess*0.7, lambda_guess * 1.3)
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


def peak_helper(all_charges, show_plot=False):
    """
    A tool to guess the location of highest peak.
    Alternatively, to show a histoagram of
    all extracted charges so guessing the pe is easier.

    Args:
        all_charges (array): Array of extracted charges
        show_plot (bool, optional): Shows a histogram of extracted charges. Defaults to False.

    Returns:
        (float): Extracted charge location of highest peak
    """
    all_charges = all_charges.ravel()
    all_charges = np.sort(all_charges)

    range_min = np.percentile(all_charges, 1)
    range_max = np.percentile(all_charges, 99)
    hist, bin_edges = np.histogram(
        all_charges, bins=100, range=(range_min, range_max), density=True
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

def initialise_lab_data(filename,args):
    """
    Initialise a FileData object, including the peak sample list
    and the baseline correction of requested.

    Args:
        filename (str): Filename of lab run
        args (args): Command-line arguments

    Returns:
        FileData: Initialised FileData object
    """    
    tel_id, run_text, n_events = get_file_info(filename)

    max_events = args.max_events

    if max_events is None:
        max_events = n_events
    else:
        max_events = min(n_events, max_events)
    
    source = SSTCAMEventSource(filename, max_events=max_events)

    for event in source:
        break
    dead_pixels_mask = event.r1.tel[tel_id].pixel_status == False

    baseline_correction, peak_indexes_all, off_pixels = get_base_peak(
        source, tel_id, max_events
    )

    if args.solo_pixels:
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
        baseline_correction
    )
    
    return dataset

def read_lab_data(
    dataset,
    args,
    max_events=None,
    extractor_params={"window_width": 12, "window_shift": 6},
    show_progress=False,
):
    """
    Perform charge extraction for a given file.

    Args:
        dataset (FileData): Initialised FileData object.
        args (argparse): Command line arguments (used for solo_pixels, fix_time_skew, and subtract_baseline)
        max_events (int, optional): Max number of events to process. Defaults to None.
        extractor_params (dict, optional): Fixed Window extraction parameters. Defaults to {"window_width": 12, "window_shift": 6}.
        show_progress (bool, optional): Show progress bar. Defaults to False.

    Returns:
        all_charges_lab: Array of extracted charges for all pixels and events
    """

    tel_id = dataset.tel_id
    source = SSTCAMEventSource(dataset.filepath, max_events=max_events)

    for event in source:
        break

    peak_index = int(np.median(dataset.peak_indexes))
    extractor_params["peak_index"] = peak_index
    peak_shifts = dataset.peak_indexes - peak_index

    image_extractor = fixed_window_extractor(source, extractor_params, correction=True)

    calib = CameraCalibrator(subarray=source.subarray, image_extractor=image_extractor)

    if show_progress:
        iterator = tqdm(
            source, total=max_events, desc="Extract charges", leave=False, flush=True
        )
    else:
        iterator = source

    charge_list = []
    for event in iterator:
        if args.fix_time_skew:
            waveforms = event.r1.tel[tel_id].waveform[0]
            new_waveforms = waveforms.copy()
            for i, pix_no in enumerate(dataset.live_pixels):
                new_waveforms[pix_no] = np.roll(waveforms[pix_no], -peak_shifts[i])
            event.r1.tel[tel_id].waveform[0] = new_waveforms
        if args.subtract_baseline:
            event.r1.tel[tel_id].waveform[0] += dataset.baseline_corrections[:, np.newaxis]
        calib(event)
        charge_list.append(event.dl1.tel[tel_id].image)
    all_charges_lab = np.vstack(charge_list).T
    all_charges_lab = all_charges_lab[dataset.live_pixels]

    return all_charges_lab


def get_bad_fit_mask(value_lists):
    """
    Generates a mask for pixels with a bad fit.
    bad_1: pe_sigma > 0.5 * pe
    bad_2: abs(pe - pe_med) > 3 * np.std(pe)
    bad_3: abs(pe_sigma - pe_s_med) > 3 * np.std(pe_sigma)

    Args:
        value_lists (dict): Dict of arrays of extracted parameters

    Returns:
        (bool array): Mask of pixels with bad fits
    """
    pe = np.array(value_lists["pe"])
    pe_sigma = np.array(value_lists["pe_sigma"])

    pe_med = np.median(pe)
    pe_s_med = np.median(pe_sigma)

    bad_1 = pe_sigma > 0.5 * pe
    bad_2 = abs(pe - pe_med) > 3 * np.std(pe)
    bad_3 = abs(pe_sigma - pe_s_med) > 3 * np.std(pe_sigma)

    return bad_1 | bad_2 | bad_3


def get_peak_valley_ratios(value_lists, fitter):
    """
    Generates an array of the ratios between the average of
    fit heights of the first and second peaks, and the height
    of the dip between them.

    Args:
        value_lists (dict): Dict of arrays of extracted parameters
        fitter (Fitter): SPE fitter

    Returns:
        (float array): List of ratios
    """

    peak_valley_ratios = []

    for ipix in range(len(fitter.pixel_arrays)):
        pixfit = fitter.pixel_arrays[ipix]
        tmp_peak_valley_ratios = []
        for illum_no in range(len(pixfit)):
            illum = pixfit[illum_no]
            fit_x = illum["fit_x"]
            fit_y = illum["fit_y"]

            eped = value_lists[f"eped_{illum_no}"][ipix]
            pe = value_lists["pe"][ipix]
            pe_1_charge = eped + pe
            pe_2_charge = eped + 2 * pe
            dip_charge = eped + pe * 1.5

            pe_1 = fit_y[np.argmin(np.abs(fit_x - pe_1_charge))]
            pe_2 = fit_y[np.argmin(np.abs(fit_x - pe_2_charge))]
            dip = fit_y[np.argmin(np.abs(fit_x - dip_charge))]

            peak_valley_ratio = ((pe_1 + pe_2) / 2) / dip
            tmp_peak_valley_ratios.append(peak_valley_ratio)
        peak_valley_ratios.append(tmp_peak_valley_ratios)

    return np.array(peak_valley_ratios).T
