# Integrating Visual Features in Multiple Hypothesis Tracking (VisualMHT)

Code and experiments for the paper: "Integrating Visual Features in Multiple Hypothesis Tracking Through Self-Supervised Learning", accepted at IEEE ISBI2026.

Abstract: *Multiple Object Tracking (MOT) is a crucial step in the automatic analysis of biological processes.
In this work, we introduce VisualMHT, a Bayesian Multiple Hypothesis Tracking algorithm integrating self-supervised visual features to leverage robust target identity and consistently track biological objects. This novel tracking algorithm achieves state-of-the-art results on the SINETRA synthetic datasets and demonstrates robust neuron tracking in the freshwater cnidarian Hydra vulgaris, highlighting the benefit of combining probabilistic motion models with learned visual representations.*

## Reproduce results

### Install

First clone the repository and submodules

```bash
$ git clone git@github.com:bia-pasteur/visual_mht.git
$ cd visual_mht
$ git submodule init
$ git submodule update
```

Install requirements (we advise to first initialize a new conda environement with Python 3.12)

```bash
$ pip install -r requirements.txt
```

Additional requirements (Icy, Fiji) are needed to reproduce some results. See the [installation guidelines](https://byotrack.readthedocs.io/en/latest/install.html#additional-requirements) of [ByoTrack](https://github.com/raphaelreme/byotrack) for a complete installation.

The experiment configuration files are using environment variables that needs to be set:
- $SINETRA: Path to the sinetra datasets (see below)
- $TRASE_IN: Path to the trase-in videos (see below)
- $ICY: path to icy.jar
- $FIJI: path to fiji executable

### Data

#### SINETRA
We use the [SINETRA](https://github.com/raphaelreme/SINETRA) synthetic datasets. It produces representative tracking data of fluorescence imaging of cells in freely-behaving animals.

It can be generated following the Sinetra guidelines. The set the $SINETRA environment variable to indicate the location of the dataset.

#### Trase-In
We use Hydra Vulgaris neuronal imaging with the TdTomato fluorophore that targets neuron nuclei from the [Trase-In](https://github.com/raphaelreme/trase-in) project. These can be downloaded from dryad at https://doi.org/10.5061/dryad.h9w0vt4q3. Unzip the data and define the $TRASE_IN environment variable to indicate the location of these Hydra videos.

For the Trase-In tracking experiment, we rely on a trained StarDist model that can be downloaded with the provided script:
```bash
bash scripts/download_trasein_stardist.sh
```

### Train our ResNet visual feature encoder
First, to generate the patch datasets from the videos, use our provided script:

```bash
$ bash scripts/generate_dataset.sh
```

Note that this also prepare validation datasets that rely on sparse segmentation labels. But for the sake of simplicity, we decided to exclude this part in our paper and do not rely on it any longer.

Then, the models for Trase-In and Sinetra can be trained through
```bash
$ bash scripts/train.sh  # Produce a model.ckpt in models/trasein and models/sinetra
```

### Run VisualMHT

Finally, you can run the tracking experiments on SINETRA:
```bash
# We use 5 random seeds: 111, 222, 333, 444 and 555
$ bash scripts/track.sh 111
$ bash scripts/track.sh 222
$ bash scripts/track.sh 333
$ bash scripts/track.sh 444
$ bash scripts/track.sh 555
```

To display the results table:
```bash
$ python scripts/aggregate_results.py
```

You can also run on our Trase-In example and manually inspect the tracks with:
```bash
$ expyrun configs/track/trasein.yml  # Requires StarDist
```
