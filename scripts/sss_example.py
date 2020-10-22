"""Example to show the influence of SSS on beamformer performance.

This script explores the effect of Signal Space Separation (SSS) on both the
rank of the covariance matrix and beamformer output.
"""

import os.path as op
import matplotlib.pyplot as plt
import mne
from mne.beamformer import make_lcmv, apply_lcmv
from mne.datasets import sample
from mne.preprocessing import maxwell_filter, find_bad_channels_maxwell
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

# first find bad channels to prevent noise spreading
raw_chans = raw.copy()
raw_chans.filter(0.1, None)
noisy_chans, flat_chans = find_bad_channels_maxwell(raw_chans,
                                                    cross_talk=crosstalk_file,
                                                    calibration=calib_file)

# Maxwell filtering
raw.info['bads'].extend(noisy_chans + flat_chans)
raw_sss = maxwell_filter(raw, cross_talk=crosstalk_file,
                         calibration=calib_file)
raw_sss.pick_types(meg='grad', eeg=False)
raw.pick_types(meg='grad', eeg=False, exclude='bads')

# filter for gamma
raw_sss.filter(60., 80.)
raw.filter(60., 80.)

# create epochs and compute the covariance
epochs_sss = mne.Epochs(raw_sss, events, event_id, tmin, tmax,
                        baseline=(None, 0), preload=True)
cov = mne.compute_covariance(epochs_sss)

# if the covariance matrix is to be plotted:
# cov.plot(epochs_sss.info)

# estimate rank:
rank_est = np.linalg.matrix_rank(cov.data)

# do an svd on the data:
sing_vals = sp.linalg.svd(cov.data, compute_uv=False)
sing_vals[sing_vals <= 0] = 1e-10 * sing_vals[sing_vals > 0].min()

# plot the singular value spectrum:
rank_col = 'red'
plt.plot(sing_vals, color='navy', linewidth=2)
plt.axvline(rank_est, color=rank_col, linestyle='--')
plt.text(75, sing_vals[3], 'rank estimate = %s' % rank_est, color=rank_col)
plt.yscale('log')
plt.ylabel('Singular values')
plt.xlabel('Singular value index')

# save the figure:
fig_fname = op.join(fig_path, 'sing_vals_sss.eps')
plt.savefig(fig_fname)
plt.show()

# epoch both the not-SSS'ed data as well:
epochs_raw = mne.Epochs(raw, events, event_id, tmin, tmax,
                        baseline=(None, 0), preload=True)

# the different configuration we want to explore:
eps = ['raw', 'sss', 'sss', 'sss']
whitening = (False, False, False, True)
subspace = (False, False, True, False)
labels = ['raw', 'sss', 'sss_and_subspace', 'sss_and_whitening']

# Loop over the different combinations:
for ep, whiten, subs, label in zip(eps, whitening, subspace, labels):

    # Set defaults:
    rank = None
    noise_cov = None
    reg = 0.05

    if ep == 'raw':
        # this is the plain MEG data without SSS
        evoked = epochs_raw.average()
        data_cov = mne.compute_covariance(epochs_raw, tmin=0.01, tmax=0.2,
                                          method='empirical')

    elif ep == 'sss':
        # this is the SSS'ed data
        evoked = epochs_sss.average()
        data_cov = mne.compute_covariance(epochs_sss, tmin=0.01, tmax=0.2,
                                          method='empirical')

        if whiten is True:
            # for whitening we need a noise covariance matrix:
            noise_cov = mne.compute_covariance(epochs_sss, tmin=tmin,
                                               tmax=-0.01, method='empirical')
        elif subs is True:
            # for subspace inversion we need to know the rank because it
            # does not get detected automatically due to the SSS
            rank = dict(grad=72)  # rank does not get estimated correctly
            reg = 0.1  # we also increase regularization a bit
    else:
        raise ValueError('This is a combination I do not know.')

    # read in the forward model
    fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-vol-7-fwd.fif'
    forward = mne.read_forward_solution(fwd_fname)

    # make LCMV beamformer with the configuration we set above:
    filters = make_lcmv(evoked.info, forward, data_cov, reg=reg,
                        noise_cov=noise_cov, pick_ori='max-power',
                        weight_norm='unit-noise-gain', rank=rank)

    # apply LCMV beamformer
    stc = apply_lcmv(evoked, filters, max_ori_out='signed')
    stc.crop(-0.1, 0.25)

    # plotting specs
    _, t_peak = stc.get_peak(tmin=0, tmax=0.2)
    t_idx = stc.time_as_index(t_peak)
    plot_max = np.max(np.abs(stc.data[:, t_idx]))  # @ peak time
    plot_min = 0.75 * plot_max
    plot_mid = plot_min + ((plot_max - plot_min) / 2)
    lims = [plot_min, plot_mid, plot_max]

    # plot the output
    kwargs = dict(src=forward['src'], subject='sample',
                  subjects_dir=subjects_dir, initial_time=np.round(t_peak, 5),
                  verbose=True)
    stc.plot(mode='stat_map', clim=dict(kind='value', pos_lims=lims),
             **kwargs)

    # save figure
    fig_fname = op.join(fig_path, 'beamformer_%s.png' % label)
    plt.savefig(fig_fname)
