import argparse
import os
import shutil

from .bulk import (
    do_fitting_spe,
    get_charge_res_output,
    get_datasets_spe,
    write_report_charge,
    write_reports_spe,
    initialise_datasets
)
from .utilities import (
    add_args_both,
    add_parser_args,
    get_both_args,
    validate_args_charge_res,
    validate_args_spe,
)


def run_both(parser, args):
    spe_args, lambda_guesses, charge_args, delete_checkpoints = get_both_args(
        parser, args=args
    )

    print("=== SPE fitting ===")

    run_spe(spe_args, lambda_guesses=lambda_guesses)

    print("=== Charge Resolution ===")

    run_charge_res(charge_args)

    if delete_checkpoints:
        print("Deleting checkpoints...")
        shutil.rmtree(os.path.join(spe_args.output_dir, "checkpoints"))
        shutil.rmtree(os.path.join(charge_args.output_dir, "checkpoints"))


def run_charge_res(args, parser=None):
    if parser is not None:
        args = validate_args_charge_res(parser, args=args)

    for remove_bad_pixels in [False, True]:
        if remove_bad_pixels:
            print("= Good pixels =")
        else:
            print("= All pixels =")
        args.remove_bad_pixels = remove_bad_pixels
        charge_datasets = initialise_datasets(args)
        df, df_2d, run_text = get_charge_res_output(charge_datasets,args)
        write_report_charge(args, df, df_2d, run_text)


def run_spe(args, lambda_guesses=None, parser=None):
    if parser is not None:
        args, lambda_guesses = validate_args_spe(parser, args=args)

    datasets = initialise_datasets(args)
    datasets = get_datasets_spe(datasets, args, lambda_guesses)
    if args.peak_helper:
        return None
    fit = do_fitting_spe(args, datasets)
    write_reports_spe(args, datasets, fit)


def main():
    parser = argparse.ArgumentParser(
        prog="sstcam",
        description="SSTCAM analysis (SPE fitting, charge resolution, or both).",
    )
    subparsers = parser.add_subparsers(
        title="subcommands",
        dest="command",
        required=True,
        help="Whether to run SPE extration, charge resolution, or both (spe/charge-res/both)",
    )

    spe_parser = subparsers.add_parser("spe", help="Run SPE fitting and make a report.")
    spe_parser = add_parser_args(spe_parser, "spe")

    cr_parser = subparsers.add_parser(
        "charge-res", help="Run charge resolution analysis and make a report."
    )
    cr_parser = add_parser_args(cr_parser, "charge_res")

    both_parser = subparsers.add_parser(
        "both", help="Run SPE fitting, then charge resolution"
    )
    both_parser = add_args_both(both_parser)

    args = parser.parse_args()

    try:
        if args.command == "spe":
            run_spe(args, parser=parser)
        elif args.command == "charge-res":
            run_charge_res(args, parser=parser)
        elif args.command == "both":
            run_both(parser, args)
    except KeyboardInterrupt:
        print("Keyboard interrupt.")


if __name__ == "__main__":
    main()
