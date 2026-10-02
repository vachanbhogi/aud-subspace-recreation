# Convolutional neural network models describe the encoding subspace of local circuits in auditory cortex 

This repository contains data and example scripts for analysis from Wingert, JC, Parida, S, Norman-Haignere, SV, David, SV. (2026)
Convolutional neural network models describe the encoding subspace of local circuits in auditory cortex. _Nature Neuroscience_. 
In press.

High-density multi-channel neurophysiology data were collected from primary (A1) 
and secondary (PEG) fields of auditory cortex of passively listening ferrets during 
presentation of a large natural sound library. Single unit spikes were sorted using 
Kilosort. This dataset includes spike times for 2128 A1 units and 746 PEG units, 
collected from 64 recording sites in four animals. Stimulus waveforms were 
transformed to log-spaced spectrograms for analsysis (32 channels, 200-20000 Hz, 10 
ms time bins). The data set includes raw sound waveforms as well (44kHz sampling).


## File contents

Neural data and model fits are each stored in one file per site. Individual 
experiments utilized different subsets of the larger natural sound library. 

### Data files
After uncompressing files `recordings.zip`, `models.zip`, and `wav.zip`,
the following should be present:
* `recordings/<SITEID>_xxxx.tgz` - Stimulus and response data for recording at site
  `<SITEID>` (PRN018a, CLT027c, etc.). The `xxxx` suffix is a hash used for internal
  organization.
* `models/CNN/<SITED>_xxx/` - Folder containing trained CNN model for site `<SITEID>`
* `models/LN/<SITED>_xxx/` - Folder containing trained LN model for site `<SITEID>`
* `models/subspace/<SITED>_xxx/` - Folder containing trained LN model for site `<SITEID>`
* `wav/seqNNNN.wav` - Raw stimulus files in wav format. Stimulus onset/offset times are tracked as "epochs" 
  labeled with the corresponding file name in the recording files. The recording files
  contain log-spaced spectrogram ("cochleagram") versions of the stimuli, pre-processed as
  they were for the model fits.
* `cell_list.csv` - list of units with cell type and depth information
* `model_performance.csv` - prediction accuracy for different models for each unit
* `neuron_pair_similarity.csv` - pairwise similarity of SSRFs between units
* `neuron_ss_tuning_pcs.csv` - SSRF symmetry properties per unit
* `ssrf_size_overlap.csv` - processed data about SSRF overlap in each recording site

### Code files

* `aud_subspace_fit_demo.ipynb` - Jupyter Lab notebook demonstrating how to train encoding models
  and perform subspace analysis on trained models
* `aud_subspace_figs.ipynb` - Notebook to regenerate figures from the paper
* `aud_ss_tools.py` - Library of helper function for plots in notebooks.

### Code dependencies 

Running the demo code requires the NEMS python library, available open source at [https://github.com/lbhb/NEMS/],
installed with Tensorflow support. A couple bits of code also require the `seaborn` and `statsmodels`. Quick install 
using Anaconda python (installed from [https://www.anaconda.com/docs/getting-started/miniconda/install]):

```
conda create -n aud_subspace python=3.9
conda activate aud_subspace
pip install /<path>/<to>/NEMS[tf]
pip install seaborn statsmodels
```

Detailed instructions for NEMS installation are available at [https://github.com/LBHB/NEMS/blob/main/README.md]. Once the libraries are installed and data uncompressed, move to the directory where 
the demo scripts are located and run Jupyter Lab in the a python environment where NEMS and other dependencies are installed.


## Sharing/Access information

Data are distributed open access. Please cite this study in any publication that uses the data:
Wingert, JC, Parida, S, Norman-Haignere, SV, David, SV. (2026)
Convolutional neural network models describe the encoding subspace of local circuits in auditory cortex. _Nature Neuroscience_. 
In press. **FINAL CITATION PENDING PUBLICATION**

