Here is a walkthrough for what I do to get `sstcam_pixel_analysis` working.

First, install these two:

```
sudo apt install git-lfs
sudo apt install cmake
```

Then install `cfitsio` (choose a sensible place for the directory) with [these instructions](https://gitlab.cta-observatory.org/cta-array-elements/sst/camera/server/sstcam-server/-/blob/develop/env/install-cfitsio.sh) (reproduced below):

```
mkdir FitsIO
cd FitsIO
wget -nc https://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio-3.47.tar.gz
tar -xvf cfitsio-3.47.tar.gz
cd cfitsio-3.47
./configure --prefix=/usr/local
make -j8
make install
```

Next, with your favourite environment management software (I use `mamba`) create an environment with `python 3.12` and `pip`:

```
mkdir qcam_analysis; cd qcam_analysis
mamba create -n qcam_analysis python=3.12 pip --yes
mamba activate qcam_analysis
```

Now we install sstcam-waveform, sstcam-telecom, and sstcam-configuration.
First I do these steps so it'll automatically authenticate, otherwise it asks for my usename and password three times each.

* Create a gitlab Personal Access Token with read_repository scope
* Create the file `~/.netrc`
* Add in this information (replace with your own)
```
machine gitlab.cta-observatory.org
login simon.lee@cta-consortium.org
password <your_token>
```
* Change permission of the file
```
chmod 600 ~/.netrc
```

Now we can install those `sstcam` packages.

(but this first one is a requirement for `sstcam-telecom`)

```
pip install 'git+https://gitlab.desy.de/corel/corel-asyncio.git'

pip install 'git+https://gitlab.cta-observatory.org/cta-array-elements/sst/camera/server/sstcam-server.git#egg=sstcam_waveform&subdirectory=sstcam-waveform'
pip install 'git+https://gitlab.cta-observatory.org/cta-array-elements/sst/camera/server/sstcam-server.git#egg=sstcam_telecom&subdirectory=sstcam-telecom'
pip install 'git+https://gitlab.cta-observatory.org/cta-array-elements/sst/camera/server/sstcam-server.git#egg=sstcam_configuration&subdirectory=sstcam-configuration'
```

Then install `ctapipe_io_sstcam`

```
git clone git@gitlab.cta-observatory.org:cta-array-elements/sst/camera/analysis/ctapipe_io_sstcam.git
cd ctapipe_io_sstcam/
pip install .
```

Now you should be able to run `sstcam-pixel-analysis --help` and, after a few seconds, it'll show you the help information!
