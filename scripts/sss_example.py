"""Example to show the influence of SSS on beamformer performance.

This script explores the effect of Signal Space Separation (SSS) on both the
rank of the covariance matrix and beamformer output.
"""

import os.path as op
import matplotlib.pyplot as plt
import mne
# from mne.beamformer import make_lcmv, apply_lcmv
from mne.datasets import sample
from mne.preprocessing import maxwell_filter  #, find_bad_channels_maxwell
import numpy as np
import scipy as sp

# we import the path to save the figures to from an extra script:
from project_settings import fig_path  # noqa

# set path and load data
data_path = sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(raw_fname, preload=True)

# sort out events
event_id = 4
tmin, tmax = -0.2, 0.5
events = mne.find_events(raw)

# SSS files
calib_file = op.join(data_path, 'SSS', 'sss_cal_mgh.dat')
crosstalk_file = op.join(data_path, 'SSS', 'ct_sparse_mgh.fif')

# Maxwell filtering
raw_sss = maxwell_filter(raw, cross_talk=crosstalk_file,
                         calibration=calib_file)

raw.info['bads'] = []  # for this demo we want to keep all channels
raw_sss.pick_types(meg=True, eeg=False)
raw.pick_types(meg=True, eeg=False)
raw.info['projs'] = []  # empty projectors, we don't need them for this demo

# create epochs and compute the covariance
epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    baseline=(None, 0), preload=True)
epochs_sss = mne.Epochs(raw_sss, events, event_id, tmin, tmax,
                        baseline=(None, 0), preload=True)
cov = mne.compute_covariance(epochs)
cov_sss = mne.compute_covariance(epochs_sss)

# if the covariance matrix is to be plotted:
# cov_sss.plot(epochs_sss.info)

# estimate rank:
rank_est = np.linalg.matrix_rank(cov.data)
rank_est_sss = np.linalg.matrix_rank(cov_sss.data)

# original data case
sing_vals = sp.linalg.svd(cov.data, compute_uv=False)
sing_vals[sing_vals <= 0] = 1e-10 * sing_vals[sing_vals > 0].min()

# plot the singular value spectrum:
rank_col = 'red'
y_lims = (10e-43, 10e-20)
plt.figure()
plt.plot(sing_vals, color='navy', linewidth=2)
plt.axvline(rank_est, color=rank_col, linestyle='--')
plt.text(200, sing_vals[3], 'rank estimate = %s' % rank_est, color=rank_col)
plt.ylim(y_lims)
plt.yscale('log')
plt.ylabel('Singular values')
plt.xlabel('Singular value index')

# save the figure:
fig_fname = op.join(fig_path, 'sing_vals_2sens.eps')
plt.savefig(fig_fname)


# SSS'ed data case
sing_vals = sp.linalg.svd(cov_sss.data, compute_uv=False)
sing_vals[sing_vals <= 0] = 1e-10 * sing_vals[sing_vals > 0].min()

# plot the singular value spectrum:
plt.figure()
plt.plot(sing_vals, color='navy', linewidth=2)
plt.axvline(rank_est_sss, color=rank_col, linestyle='--')
plt.text(75, sing_vals[3], 'rank estimate = %s' % rank_est_sss, color=rank_col)
plt.ylim(y_lims)
plt.yscale('log')
plt.ylabel('Singular values')
plt.xlabel('Singular value index')

# save the figure:
fig_fname = op.join(fig_path, 'sing_vals_sss.eps')
plt.savefig(fig_fname)
