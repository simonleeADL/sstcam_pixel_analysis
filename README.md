# sstcam_pixel_analysis
This tools allows you to take lab data from an SSTCAM and perform Single Photoelectron (SPE) spectrum fitting and charge resolution plotting. The results are then readable in HTML reports with interactive plots.

It is build atop [`spefit`](https://gitlab.cta-observatory.org/cta-consortium/aswg/tools/spefit) as its foundation.

This code is still in early development, and I have much to learn before it's ready for widespread use.

## Installation
Clone, and when in the directory, install with `pip install .`

### Requirements

The following requirements should be installed first manually. The rest of the requirements, which are installed automatically, can be found in `pyproject.toml`.

* [`git-lfs`](https://git-lfs.github.com/)
* [`cmake`](https://cmake.org/)
* [`pip`](https://pypi.org/project/pip/)
* [`cfitsio`](https://heasarc.gsfc.nasa.gov/fitsio/) ([see install script](https://gitlab.cta-observatory.org/cta-array-elements/sst/camera/server/sstcam-server/-/blob/develop/env/install-cfitsio.sh))
* [`corel-asyncio`](https://gitlab.desy.de/corel/corel-asyncio)

Required for:

* [`sstcam-waveform`](https://gitlab.cta-observatory.org/cta-array-elements/sst/camera/server/sstcam-server/-/tree/develop/sstcam-waveform)
* [`sstcam-telecom`](https://gitlab.cta-observatory.org/cta-array-elements/sst/camera/server/sstcam-server/-/tree/develop/sstcam-telecom)
* [`sstcam-configuration`](https://gitlab.cta-observatory.org/cta-array-elements/sst/camera/server/sstcam-server/-/tree/develop/sstcam-configuration)

Required for:

* [`ctapipe_io_sstcam`](https://gitlab.cta-observatory.org/cta-array-elements/sst/camera/analysis/ctapipe_io_sstcam)

Then also:

* [`spefit`](https://gitlab.cta-observatory.org/cta-consortium/aswg/tools/spefit)
```
pip install 'git+https://gitlab.cta-observatory.org/cta-consortium/aswg/tools/spefit.git#egg=spefit'
```

## Usage
For SPE extraction, it is expected that there is a folder in which R1 `.tio` files reside of low-illumination lab runs. A `table.csv` in the same folder is required if you plan to point to the whole folder and not just individual runs.

For charge resolution calculation, it is expected that there is a folder in which R1 `.tio` files reside from a lab dynamic range scan. A `table.csv` must reside in the same folder, so that expected photons and NSB information can be referenced.

To perform SPE extraction and charge-resolution calculation, run the terminal command:

```
sstcam-pixel-analysis both /path/to/spe /path/to/dynamic_range -o /path/to/output
```

The processing runs in parallel, and unfortunately cannot show a progress bar. On my laptop the SPE extraction takes roughly 15 minutes.

The following options are also available:

* `-p` is the initial guess for extracted photoelectron charge (default automatically choosen depending on window width)
* `-w` is the number of samples you want to use for the extraction (defaults to 16)
* `--leave_time_skew` to *not* choose extraction peaks on a per-pixel basis
* `--leave_baseline` to *not* perform a rudimentary background subtraction based on the average of the first 5 samples (does not do much)
* `--peak_helper` to plot the SPE distributions across all pixels to get an idea of p.e. size
* `--overwrite` which ignore previous extracted charge checkpoints in the same output directory
* `--max_events` to process only a certain number of events per run (good for quickly testing)

Also available as individual functions are:
```
sstcam-pixel-analysis spe
```
and
```
sstcam-pixel-analysis charge-res
```
with syntax available with `--help`.
