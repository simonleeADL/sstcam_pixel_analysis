import argparse
import csv
import itertools
import json
import os
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from ctapipe.io import EventSource
from astropy.io import fits
from scipy.interpolate import interp1d
from sstcam_configuration.mapping.pixel import MappingPixel

from ctapipe_io_sstcam import SSTCAMEventSource

PIX_MAP = MappingPixel.from_version("v0.0.1-sst")

SCRIPT_DIR = Path(__file__).resolve().parent

with open(f"{SCRIPT_DIR}/config.yaml", "r") as _f:
    _cfg = yaml.safe_load(_f)

PDE = _cfg["PDE"]


def spin(stop, start_time, n_files):
    """
    Animate a spinning

    Args:
        stop (bool): Bool for when to stop the animation.
        start_time (time): Timestamp when processing started
        n_files (int): Number of files being processed
    """
    for c in itertools.cycle("|/-\\"):
        if stop():
            break
        elapsed = int(time.time() - start_time)
        mins, secs = divmod(elapsed, 60)
        sys.stdout.write(
            f"\rParallel processing {n_files} files... {c} {mins:02d}:{secs:02d}"
        )
        sys.stdout.flush()
        time.sleep(0.2)
    total_elapsed = int(time.time() - start_time)
    mins, secs = divmod(total_elapsed, 60)
    sys.stdout.write(f"\rProcessing done in {mins:02d}:{secs:02d}{' '*20}\n")
    sys.stdout.flush()


def format_hz(hz):
    """
    Format a hz value in to something human-readable.

    Args:
        hz (float): A number of Hz unit

    Returns:
        str: Formatted with appropriate digits and unit
    """
    if hz == 0:
        return "0 Hz"

    units = [("Hz", 1), ("kHz", 1e3), ("MHz", 1e6), ("GHz", 1e9)]

    for unit, scale in reversed(units):
        scaled = hz / scale
        if scaled >= 10:
            rounded = round(scaled)
            if 10 <= rounded <= 9999:
                return f"{rounded} {unit}"

    return f"{round(hz)} Hz"

def too_bright(lk,f):
    filepath = Path(f)
    n_ph = lk["n_ph"][filepath]
    if n_ph > 1200:
        return True
    return False

def baseline_subtract(event, tel_id=1):
    w = event.r1.tel[tel_id].waveform[0]
    idx = np.r_[0:20, w.shape[1] - 40 : w.shape[1]]
    baseline = w[:, idx].mean(axis=1)
    event.r1.tel[tel_id].waveform[0] -= baseline[:, None]
    return event


def load_calibration(filename):
    with open(filename) as f:
        data = json.load(f)
    transfer_function = {}
    for w, pix_dict in data.items():
        transfer_function[w] = {}
        for pix_id, d in pix_dict.items():
            extracted_adc = np.array(d["extracted_adc"])
            adc_per_pe = np.array(d["adc_per_pe"])
            # Clamp values outside the lab range
            f_interp = interp1d(
                extracted_adc,
                adc_per_pe,
                bounds_error=False,
                fill_value=(adc_per_pe[0], adc_per_pe[-1]),
            )
            transfer_function[w][int(pix_id)] = f_interp
    return transfer_function

def get_lookup_charge_res(args):
    base_dir = args.input_dir
    lk = pd.read_csv(Path(base_dir) / "table.csv")
    lk["filename"] = lk["name"].apply(lambda s: Path(base_dir) / Path(s).name)
    lk = lk.set_index("filename")[["n_ph", "n_ph_meas", "nsb_meas", "nsb"]]
    return lk

def calibrate_extracted_adc(extracted_adc, pix_id, transfer_function, w):
    w = str(w)
    if w not in transfer_function:
        return extracted_adc
    if pix_id not in transfer_function[w]:
        return extracted_adc
    f_interp = transfer_function[w][pix_id]
    calibrated = extracted_adc / f_interp(extracted_adc)
    return calibrated


def limit_curves(n, nsb, t_w, n_e, sigma_g, enf, pde):
    """
    Equation for calculating the Goal and Requirement curves, as defined
    in SCI-MC/121113.
    https://portal.cta-observatory.org/recordscentre/Records/SCI/
    SCI-MC/measurment_errors_system_performance_1YQCBC.pdf

    Parameters
    ----------
    n : ndarray
        Number of photons (variable).
    nsb : float
        Number of NSB photons.
    t_w : float
        Effective signal readout window size.
    n_e : float
        Electronic noise
    sigma_g : float
        Multiplicative errors of the gain.
    enf : float
        Excess noise factor.
    """
    q = n * pde
    sigma_0 = np.sqrt(nsb * t_w + n_e**2)
    sigma_enf = 1 + enf
    sigma_q = np.sqrt(sigma_0**2 + sigma_enf**2 * q + sigma_g**2 * q**2)
    return sigma_q / q


def requirement(n,nsb=0.125):
    """
    CTA requirement curve.

    Parameters
    ----------
    n : ndarray
        Number of photons
    nsb : float
        Night Sky Background rate (p.e./ns)
    """
    nsb+=0.005
    t_w = 15
    n_e = 0.87
    sigma_g = 0.1
    enf = 0.2
    requirement = limit_curves(n, nsb, t_w, n_e, sigma_g, enf, PDE)

    max_photons = 4000
    min_photons = 4
    requirement[(n > max_photons) & (n < min_photons)] = np.nan

    return requirement


def get_pixel_info(pixel_index: int, pix_map: "MappingPixel" = PIX_MAP):
    """
    Print detailed mapping information for a given pixel index in a readable format.

    Args:
        pixel_index (int): Index of the pixel.
        pix_map (MappingPixel, optional): MappingPixel object (default is PIX_MAP).

    Returns:
        slot (int): Pixel slot in the camera
        asic (int): Pixel ASIC in the slot
        asic_ch (int): Pixel channel in the ASIC
    """

    mask = pix_map.index == pixel_index
    if not mask.any():
        print(f"Pixel {pixel_index} not found in the mapping.")
        return

    slot = pix_map.module[mask].item()
    asic = pix_map.asic[mask].item()
    asic_ch = pix_map.channel[mask].item()

    return slot, asic, asic_ch


def blank_values(n_files):
    """
    Creates a blank dict of relevant parameters
    for per-pixel SPE fitting depending on
    how many runs are being processed.

    Args:
        n_files (int): Number of files being processed

    Returns:
        value_lists (dict): Empty dict for SPE fit parameters
    """
    value_lists = {
        "eped_sigma": [],
        "pe": [],
        "pe_sigma": [],
        "opct": [],
        "p_value": [],
        "reduced_chi2": [],
    }
    for i in range(n_files):
        value_lists[f"lambda_{i}"] = []
    for i in range(n_files):
        value_lists[f"eped_{i}"] = []
    return value_lists


def fmt(xx, sf=2):
    """
    Formats a number to a string with 2 (or other) decimal places

    Args:
        xx (float): A number
        sf (int, optional): Number of decimal places to round to

    Returns:
        (string): Formatted number as a string
    """
    return f"{xx:.{sf}f}"


def output_filenames(in_file, out_dir, report="SPE"):
    """
    Gives some useful filenames given some
    input file and output directory

    Args:
        in_file (str): Input run file
        out_dir (str): Output directory

    Returns:
        all_charges_file: Checkpoint file of extracted charges
        report_filename: Output HTML report filename
        output_dir: The output directory (defaults to input dir)
    """
    input_stem = Path(in_file).stem
    if out_dir:
        output_dir = out_dir
    else:
        output_dir = Path(in_file).parent
    all_charges_file = f"{output_dir}/checkpoints/{input_stem}_checkpoint.npz"
    report_filename = f"{output_dir}/reports/{input_stem}_{report}.html"
    return all_charges_file, report_filename, output_dir


def get_value_lists(fitter, n_files):
    """
    Returns a dict, where each key is a different
    SPE fit parameter, each having a list of values
    corresponding to all the pixels that were fit.

    Args:
        fitter (Fitter): The SPe fitter
        n_files (int): No. files

    Returns:
        (dict): Dict of lists of SPE fit parameter values
    """
    value_lists = blank_values(n_files)

    for ipix in range(len(fitter.pixel_values)):
        for param, value in fitter.pixel_values[ipix].items():
            value_lists[param].append(value)
        for param, value in fitter.pixel_scores[ipix].items():
            if param == "chi2":
                continue
            value_lists[param].append(value)

    return value_lists


def write_pixel_spe_table(fit, live_pixels, csv_output):
    """
    Write extracted SPE parameters to file

    Args:
        fitter (Fitter): SPE fitter
        live_pixels (array): List of pixel IDs of not-dead pixels
        csv_output (str): Filename of output csv
    """
    fitter = fit.fitter
    value_lists = fit.value_lists
    peak_valleys = fit.peak_valley
    ever_bad_fit = ~np.all(fit.good_fit_masks, axis=0)

    keys = value_lists.keys()

    values = fitter.pixel_values
    errors = fitter.pixel_errors
    scores = fitter.pixel_scores

    header = ["pixel_no", "slot", "asic", "asic_ch", "good_fit"]
    for k in keys:
        if k in value_lists:
            header.append(k)
            if k in errors[0].keys():
                header.append(f"{k}_error")
    for i in range(len(peak_valleys)):
        header.append(f"peak_valley_{i}")

    with open(csv_output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for ipix, pixel_no in enumerate(live_pixels):

            slot, asic, asic_ch = get_pixel_info(pixel_no)
            row = [pixel_no, slot, asic, asic_ch]
            row.append(not ever_bad_fit[ipix])

            for k in keys:
                if k in values[ipix].keys():
                    row.append(values[ipix][k])
                    row.append(errors[ipix][k])
                else:
                    row.append(scores[ipix][k])
            for peak_valley in peak_valleys:
                row.append(peak_valley[ipix])

            writer.writerow(row)

def get_source(filename,max_events):
    if filename.endswith('.tio'):
        return SSTCAMEventSource(filename, max_events=max_events), False
    elif 'simtel' in filename:
        return EventSource(filename, max_events=max_events), True


def get_file_info(filename):
    """
    Generates text with info about the file.

    Args:
        filename (str): Run filename

    Returns:
        int: Telescope ID
        str: Text with file info
        int: Number of events in the file
    """
    if filename.endswith('.tio'):
        with fits.open(filename) as f:
            n_events = f[1].header["NAXIS2"]
            date_obs = f[0].header["DATE-OBS"]
        source = SSTCAMEventSource(filename)
        tel_id = source.tel_id
    else:
        source = EventSource(filename)
        n_events = 50000
        #for _ in source:
        #    n_events += 1
        date_obs = None
        tel_id = 1  

    run_text = ""

    run_text += f"<b>File name</b>: {Path(filename).stem}<br>"
    run_text += f"<b>File path</b>: {Path(filename).resolve()}<br>"
    run_text += f"<b>Obs date</b>: {date_obs}<br>"
    run_text += f"<b>No. events</b>: {n_events}<br>"

    return tel_id, run_text, n_events


def get_run_text_chres(filename):
    """
    Generate text to describe a charge resolution run

    Args:
        filename (str): File name of the run

    Returns:
        str: Text to describe a charge resolution run
    """
    with fits.open(filename) as f:
        n_events = f[1].header["NAXIS2"]
        date_obs = f[0].header["DATE-OBS"]

    run_text = ""
    run_text += f"<b>Files directory</b>: {Path(filename).resolve().parent}<br>"
    run_text += f"<b>Obs date (first file)</b>: {date_obs}<br>"
    run_text += f"<b>No. events (first file)</b>: {n_events}<br>"

    return run_text


def get_processing_text(args, mode="SPE"):
    """
    Generates text about the SPE processing.

    Args:
        args (args): Command-line arguments

    Returns:
        str: Text about the SPE processing.
    """
    processing_text = ""
    if mode == "SPE":
        processing_text += f"<b>No. runs used:</b> {len(args.input_file)}<br>"
        processing_text += f"<b>pe_guess:</b> {args.pe_guess}<br>"
    processing_text += f"<b>max_events:</b> {args.max_events}<br>"
    processing_text += f"<b>Extraction window width:</b> {not args.window_width}<br>"
    processing_text += f"<b>Time skew adjusted:</b> {not args.leave_time_skew}<br>"
    processing_text += f"<b>Baseline subtraction:</b> {not args.leave_baseline}<br>"
    if mode == "res":
        processing_text += (
            f"<b>Reference SPE extraction:</b> {Path(args.ref_spe).resolve()}<br>"
        )
    return processing_text


def validate_args_shared(parser, args, doing_both=False):
    """
    Validate command-line arguments that are shared betwee SPE and charge resolution

    Args:
        parser: Argument parser
        args: Command-line arguments
        doing_both (bool): If both SPE and charge res are being processed (optional, defaults to False)

    Returns:
        args: Command-line arguments
    """

    if args.output_dir:
        args.output_dir = os.path.abspath(args.output_dir).rstrip("/")
        if not os.path.isdir(args.output_dir):
            response = (
                input(
                    f"The chosen --output_dir does not exist:\n  {args.output_dir}\n"
                    f"Do you want to create it? (y/n): "
                )
                .strip()
                .lower()
            )
            if response == "y":
                os.makedirs(args.output_dir, exist_ok=True)
                print(f"Created directory: {args.output_dir}")
            else:
                parser.error(
                    f"The chosen --output_dir does not exist: {args.output_dir}"
                )
    return args


def validate_args_charge_res(parser, args=None, doing_both=False):
    """
    Validate command-line arguments for charge resolution calulation

    Args:
        parser: Argument parser
        args: Command-line arguments (optional, used in sstcam_spe_charge_res)
        doing_both (bool): If both SPE and charge res are being processed (optional, defaults to False)

    Returns:
        args: Command-line arguments
    """
    if args is None:
        args = parser.parse_args()

    args = validate_args_shared(parser, args, doing_both=doing_both)

    input_files = glob(f"{args.input_dir}*.tio")
    if len(input_files) == 0:
        parser.error(f"No .tio files found in input directory: {args.input_dir}")

    if not doing_both:
        if not os.path.isfile(args.ref_spe):
            parser.error(f"Reference SPE output not found: {args.ref_spe}")

    for f in input_files:
        all_charges_file, _, _ = output_filenames(f, args.output_dir)
        checkpoint_exists = os.path.isfile(all_charges_file)

        if checkpoint_exists and args.overwrite:
            print(f"Warning: Will overwrite dynamic range checkpoint data.")
            break

    return args


def width_to_pe_guess(w):
    if w > 26:
        return 24

    mapping = {
        2: 3,
        4: 7,
        6: 12,
        8: 14,
        10: 16,
        12: 18,
        14: 19,
        16: 20,
        18: 21,
        20: 22,
        22: 23,
        24: 23,
        26: 24,
    }

    return mapping.get(w)


def validate_args_spe(parser, args=None, doing_both=False):
    """
    Validate command-line arguments for SPE fitting

    Args:
        parser: Argument parser
        args: Command-line arguments (optional, used in sstcam_spe_charge_res)
        doing_both (bool): If both SPE and charge res are being processed (optional, defaults to False)

    Returns:
        args: Command-line arguments
    """

    if args is None:
        args = parser.parse_args()

    if not args.input_file:
        parser.error("The --input_file must be specified.")

    args = validate_args_shared(parser, args, doing_both=doing_both)

    if args.window_width % 2 != 0 or args.window_width <= 0:
        parser.error("The --window_width must be a positive even number")

    auto_pe_guess = width_to_pe_guess(args.window_width)

    if args.pe_guess is None:
        args.pe_guess = auto_pe_guess
    if args.pe_guess <= 0:
        parser.error("The --pe_guess must be a positive number")
    if abs(auto_pe_guess - args.pe_guess) > 3:
        print(
            f"Warning: Estimated p.e. guess (for QCAM) is {auto_pe_guess}, your guess is {args.pe_guess}"
        )

    lambda_guesses = None
    if os.path.isdir(args.input_file[0]):
        input_table_filename = f"{args.input_file[0]}/table.csv"
        if not os.path.isfile(input_table_filename):
            parser.error(
                "--input_file given as directory, but table.csv does not exist"
            )
        input_table = pd.read_csv(input_table_filename)
        input_files_list = []
        for full_path in input_table["name"]:
            filename = os.path.basename(full_path)
            new_path = os.path.join(args.input_file[0], filename)
            input_files_list.append(new_path)
        lambda_guesses = []
        for n_ph in input_table["n_ph"]:
            lambda_guesses.append(n_ph / 2)
        args.input_file = input_files_list

    for f in args.input_file:
        if not f.endswith(".tio") and "simtel" not in str(f):
            parser.error("The --input_file must be a .tio or simtel file.")
        if not os.path.isfile(f):
            parser.error(f"The --input_file does not exist: {f}")

    n_files = len(args.input_file)
    n_checkpoints = 0

    for f in args.input_file:
        all_charges_file, _, _ = output_filenames(f, args.output_dir)
        if os.path.isfile(all_charges_file):
            n_checkpoints += 1

    if args.overwrite:
        if n_checkpoints > 0:
            print(
                f"Warning: Will overwrite SPE checkpoint data for {n_checkpoints} files"
            )
        print(f"Processing {n_files} files")
    elif n_files - n_checkpoints > 0 and n_checkpoints > 0:
        print(
            f"Reading {n_checkpoints} checkpoints and processing {n_files-n_checkpoints} files"
        )
    elif n_checkpoints > 0:
        print(f"Reading {n_checkpoints} checkpoints")
    elif n_files > 0:
        print(f"Processing {n_files} files")

    return args, lambda_guesses


def add_parser_args(parser, report):
    """
    Add appropriate argumnets to a command-line parser.

    Args:
        parser (argparse.parser): Command-line argument parser
        report (str): Type of report ('spe' or 'charge_res')

    Returns:
        parser: Command-line argument parser
    """

    if report == "spe":
        parser.add_argument(
            "input_file",
            type=str,
            nargs="+",
            help="Path to the input data file (or multiple files)",
        )
    if report == "charge_res":
        parser.add_argument(
            "-i",
            "--input_dir",
            type=str,
            default="./",
            help="Path to input directory (optional, defaults to current directory)",
        )
        parser.add_argument(
            "-r",
            "--ref_spe",
            type=str,
            help="Path to reference .csv with extracted SPE values",
        )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="",
        help="Path to output directory (optional)",
    )
    if report == "spe":
        parser.add_argument(
            "-p",
            "--pe_guess",
            type=float,
            help="Initial guess for extracted PE size (default will be automated)",
        )
        parser.add_argument(
            "--peak_helper", action="store_true", help="Show peak finding helper"
        )
    parser.add_argument(
        "-w",
        "--window_width",
        type=int,
        default=16,
        help="Sample window width for extractor, must be even (default: 16)",
    )
    parser.add_argument(
        "--max_events", type=int, default=None, help="Maximum events to process per run"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the extracted charges checkpoint (if it exists)",
    )
    parser.add_argument(
        "--leave_time_skew",
        action="store_true",
        help="Do not do time skew adjustment",
    )
    parser.add_argument(
        "--leave_baseline",
        action="store_true",
        help="Do not do rudimentary baseline subtraction",
    )
    return parser


def add_args_both(parser):
    """
    Add arguments for both SPE fitting and
    charge resolution when run at the same time.

    Args:
        parser (argparse.parser): Command-line argument parser

    Returns:
        parser (argparse.parser): Command-line argument parser
    """
    parser.add_argument(
        "input_spe_dir", help="Path to the SPE input data file(s) or directory."
    )
    parser.add_argument(
        "input_charge_res_dir",
        help="Path to the Charge Resolution input data directory.",
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, help="Path to the base output directory."
    )
    parser.add_argument(
        "--delete_checkpoints",
        action="store_true",
        help="Delete checkpoint files at the end.",
    )

    temp_spe = argparse.ArgumentParser(add_help=False)
    add_parser_args(temp_spe, "spe")
    for action in temp_spe._actions:
        if not action.option_strings:
            continue
        if any(
            flag in ("-o", "--output_dir", "-i", "--input_dir", "--ref_spe")
            for flag in action.option_strings
        ):
            continue
        parser._add_action(action)

    temp_charge = argparse.ArgumentParser(add_help=False)
    add_parser_args(temp_charge, "charge_res")
    existing_flags = {flag for a in parser._actions for flag in a.option_strings}
    for action in temp_charge._actions:
        if not action.option_strings:
            continue
        if any(flag in existing_flags for flag in action.option_strings):
            continue
        if any(
            flag
            in ("-o", "--output_dir", "--peak_helper", "-i", "--input_dir", "--ref_spe")
            for flag in action.option_strings
        ):
            continue
        parser._add_action(action)
    return parser


def get_both_args(parser, args=None):
    """
    Get args for both SPE and charge res

    Args:
        parser (argparse.parser): Command-line argument parser
        args (args, optional): Command-line arguments. Defaults to None.

    Returns:
        args: Arguments for SPE
        lambda_guesses: Initial guesses for lambda
        args: Arguments for charge resolution
        bool: Whether to delete checkpoint data once done
    """
    if args is None:
        args = parser.parse_args()

    args = validate_args_shared(parser, args)

    # Ensure output subdirectories exist
    spe_output_dir = os.path.join(args.output_dir, "spe")
    charge_output_dir = os.path.join(args.output_dir, "charge_res")
    os.makedirs(spe_output_dir, exist_ok=True)
    os.makedirs(charge_output_dir, exist_ok=True)

    spe_args = argparse.Namespace(**vars(args))
    spe_args.input_file = [args.input_spe_dir]
    spe_args.output_dir = spe_output_dir
    spe_args, lambda_guesses = validate_args_spe(parser, args=spe_args, doing_both=True)

    charge_args = argparse.Namespace(**vars(args))
    charge_args.input_dir = args.input_charge_res_dir
    charge_args.output_dir = charge_output_dir
    charge_args.ref_spe = os.path.join(spe_output_dir, "spe_output.csv")
    charge_args = validate_args_charge_res(parser, args=charge_args, doing_both=True)

    return spe_args, lambda_guesses, charge_args, args.delete_checkpoints
