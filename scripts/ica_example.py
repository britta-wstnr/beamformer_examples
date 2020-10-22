"""Example to show the influence of ICA on covariance matrix rank."""

import os.path as op
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample
from mne.preprocessing import ICA
import numpy as np
import scipy as sp

# we import the path to save the figures to from an extra script:
from project_settings import fig_path  # noqa

# set path and load data
data_path = sample.data_path()
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(raw_fname, preload=True)

# define events
event_id = 4
tmin, tmax = -0.2, 0.5
events = mne.find_events(raw)

# pick only gradiometers for this demo
raw.pick_types(meg='grad', eeg=False)

# high-pass filter the data for ICA
filt_raw = raw.copy()
filt_raw.load_data().filter(l_freq=1., h_freq=None)

# run ICA
ica = ICA(n_components=15, random_state=510)
ica.fit(filt_raw)

# if desired: plot components
# ica.plot_components()

# we throw out three components:
ica.exclude = [0, 1, 3]

# exclude those components from the data:
raw_ica = raw.copy()
ica.apply(raw_ica)

# epoch the data and compute the covariance
epochs = mne.Epochs(raw_ica, events, event_id, tmin, tmax, baseline=(None, 0),
                    preload=True)
cov = mne.compute_covariance(epochs)

# if the covariance matrix is to be plotted:
# cov.plot(epochs.info)

# estimate rank:
rank_est = np.linalg.matrix_rank(cov.data)

# do an svd on the data:
sing_vals = sp.linalg.svd(cov.data, compute_uv=False)
sing_vals[sing_vals <= 0] = 1e-10 * sing_vals[sing_vals > 0].min()

# plot singular value spectrum and annotate:
rank_col = 'red'
plt.plot(sing_vals, color='navy', linewidth=2)
plt.axvline(rank_est, color=rank_col, linestyle='--')
plt.text(165, sing_vals[3], 'rank estimate = %s' % rank_est, color=rank_col)
plt.yscale('log')
plt.ylabel('Singular values')
plt.xlabel('Singular value index')

# save the figure
fig_fname = op.join(fig_path, 'sing_vals_ica.eps')
plt.savefig(fig_fname)
plt.show()
