# Trial-level sequence predictions

Per-trial classifier probabilities for the **fast-sequence trials** of the
Highspeed dataset (Wittkuhn & Schuck 2021,[Nature Communications](https://www.nature.com/articles/s41467-021-21970-2)) plus new MEG dataset
exported from both modalities:

- **MEG** — 30 subjects, 4 inter-stimulus intervals (32, 64, 128, 512 ms),
  100 Hz time axis, 5 image classes.
- **fMRI (3T)** — up to 40 subjects, 5 inter-stimulus intervals (32, 64, 128,
  512, 2048 ms), 13 TRs (1.25 s each) per trial, 5 image classes.

In every trial the participant saw an ordered sequence of 5 images drawn from
the 5 classes `['face', 'house', 'cat', 'shoe', 'chair']` (Haxby 2001
dataset). The per-trial probability matrix and the corresponding stimulus
order are all you need to reproduce sequence-detection analyses such as
**TDLM** or **SODA**.

Examples for loading the data with Python, R, Julia, MATLAB are below.

See also (FASTIMAGES-benchmark/examples)[https://github.com/CIMH-Clinical-Psychology/FASTIMAGES-benchmark/tree/main/examples] for example scripts

## Layout

```
sequence_predictions/
├── MEG/   sub-01.h5 … sub-30.h5
└── fMRI/  sub-01.h5 … sub-40.h5
```

One HDF5 file per subject, gzip-compressed.

## File contents

Each file contains five datasets and a handful of file-level attributes.

| dataset           | shape                         | dtype   | description                                        |
| ----------------- | ----------------------------- | ------- | -------------------------------------------------- |
| `probas`          | `(n_trials, n_timepoints, 5)` | float32 | classifier probabilities, last axis = `categories` |
| `sequences`       | `(n_trials, 5)`               | int16   | class indices (0..4) in presentation order         |
| `sequence_labels` | `(n_trials, 5)`               | str     | same, but as labels (`'face'`, `'house'`, …)       |
| `intervals`       | `(n_trials,)`                 | int32   | inter-stimulus interval per trial, in ms           |
| `trial_ids`       | `(n_trials,)`                 | int32   | original trial number from the experiment          |

File-level attributes (`f.attrs`):

| attribute      | type  | always | description                                      |
| -------------- | ----- | ------ | ------------------------------------------------ |
| `modality`     | str   | yes    | `'MEG'` or `'fMRI'`                              |
| `subject`      | str   | yes    | e.g. `'sub-01'`                                  |
| `categories`   | list  | yes    | column order of the last axis of `probas`        |
| `description`  | str   | yes    | free-text description of the time axis           |
| `sfreq`        | int   | MEG    | sampling rate in Hz (100)                        |
| `tmin`, `tmax` | float | MEG    | epoch start/end in seconds, relative to text cue |
| `tr_duration`  | float | fMRI   | TR length in seconds (1.25)                      |
| `n_trs`        | int   | fMRI   | number of TRs per trial (13)                     |

### Time axis conventions

- **MEG**: epochs start `tmin = 3.1 s` after the text cue, i.e. **200 ms
  before** the onset of the first sequence image. With `sfreq = 100 Hz` the
  k-th sample corresponds to `−200 + 10·k` ms relative to the first image.
- **fMRI**: the 13 TRs are aligned to the sequence onset and ordered by time
  (TR index 1 = the TR containing or immediately after sequence onset).

### Class column order

The probability columns of `probas[..., :]` follow the `categories`
attribute, i.e. column 0 = `'face'`, 1 = `'house'`, 2 = `'cat'`,
3 = `'shoe'`, 4 = `'chair'`. The same mapping is used for `sequences`.

### Normalization

The probabilities are raw classifier outputs (rows sum to 1). The reference
analyses (Wittkuhn 2021) divide each class column by its mean across time
per trial — `proba / proba.mean(axis=0)` — to remove class bias before
running TDLM/SODA. Apply this yourself if you want to reproduce those
results.

---

## Loading the data

The examples below open `MEG/sub-01.h5` and read out the probability matrix,
the sequence of each trial, and the file attributes.

### Python (h5py)

```python
import h5py

with h5py.File('MEG/sub-01.h5', 'r') as f:
    probas    = f['probas'][:]                          # (n_trials, T, 5)
    sequences = f['sequences'][:]                       # (n_trials, 5)
    labels    = f['sequence_labels'][:].astype(str)
    intervals = f['intervals'][:]                       # (n_trials,)
    categories = list(f.attrs['categories'])
    sfreq      = float(f.attrs['sfreq'])

print(probas.shape, sequences.shape, categories)
```

### Python (pandas → long DataFrame)

```python
import h5py, numpy as np, pandas as pd

with h5py.File('MEG/sub-01.h5', 'r') as f:
    probas    = f['probas'][:]
    sequences = f['sequences'][:]
    intervals = f['intervals'][:]
    cats      = list(f.attrs['categories'])
    sfreq     = float(f.attrs['sfreq'])

n_trials, n_t, n_cls = probas.shape
time_ms = np.arange(n_t) * (1000 / sfreq) - 200          # MEG convention

df = (pd.DataFrame(
        probas.reshape(-1, n_cls),
        columns=cats)
      .assign(trial=np.repeat(np.arange(n_trials), n_t),
              time_ms=np.tile(time_ms, n_trials),
              interval=np.repeat(intervals, n_t))
      .melt(id_vars=['trial', 'time_ms', 'interval'],
            var_name='class', value_name='probability'))
```

### R (hdf5r)

```r
library(hdf5r)

f <- H5File$new("MEG/sub-01.h5", mode = "r")

probas      <- f[["probas"]]$read()         # array, dim = (5, T, n_trials) in R
sequences   <- f[["sequences"]]$read()      # matrix, dim = (5, n_trials)
intervals   <- f[["intervals"]]$read()
categories  <- h5attr(f, "categories")
sfreq       <- h5attr(f, "sfreq")

f$close_all()
# NOTE: R reverses HDF5 dimension order vs. Python.
# To match the documented shape (n_trials, T, 5) use:
probas <- aperm(probas, c(3, 2, 1))
```

The `rhdf5` package (Bioconductor) works similarly via `h5read()`.

### MATLAB

```matlab
fname = 'MEG/sub-01.h5';

probas     = h5read(fname, '/probas');        % (5, T, n_trials), reversed dims
sequences  = h5read(fname, '/sequences');     % (5, n_trials)
intervals  = h5read(fname, '/intervals');
categories = h5readatt(fname, '/', 'categories');
sfreq      = h5readatt(fname, '/', 'sfreq');

% match the documented shape (n_trials, T, 5):
probas = permute(probas, [3 2 1]);
```

Use `h5disp(fname)` to print the full file structure.

### Julia (HDF5.jl)

```julia
using HDF5

h5open("MEG/sub-01.h5", "r") do f
    probas     = read(f["probas"])         # (5, T, n_trials), reversed dims
    sequences  = read(f["sequences"])
    intervals  = read(f["intervals"])
    categories = read_attribute(f, "categories")
    sfreq      = read_attribute(f, "sfreq")

    # match the documented shape (n_trials, T, 5):
    probas = permutedims(probas, (3, 2, 1))
end
```
