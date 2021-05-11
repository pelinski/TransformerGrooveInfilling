import sys

sys.path.append('../')

from hvo_sequence.hvo_seq import HVO_Sequence
from hvo_sequence.io_helpers import get_grid_position_and_utiming_in_hvo

import numpy as np
import librosa
import scipy.signal
import resampy
import soundfile as psf
import warnings
import torch


def read_audio(filepath, sr=None, mono=True, peak_norm=False):
    """
    Read audio
    @param filepath: str
    @param sr: int
    @param mono: boolean
    @param peak_norm: boolean
    @returns y: list
    """
    try:
        y, _sr = psf.read(filepath)
        y = y.T
    except RuntimeError:
        y, _sr = librosa.load(filepath, mono=False, sr=None)

    if sr is not None and sr != _sr:
        y = resampy.resample(y, _sr, sr, filter='kaiser_fast')
    else:
        sr = _sr

    if mono:
        y = librosa.to_mono(y)

    if peak_norm:
        y /= np.max(np.abs(y))

    return y, sr


def cq_matrix(n_bins_per_octave, n_bins, f_min, n_fft, sr):
    """
    ConstantQ Transform matrix with triangular log-spaced filterbank.
    @param n_bins_per_octave: int
    @param n_bins: int
    @param f_min: float
    @param n_fft: int
    @param sr: int
    @returns c_mat: matrix
    @returns: f_cq: list (triangular filters center frequencies)
    """
    # note range goes from -1 to bpo*num_oct for boundary issues
    f_cq = f_min * 2 ** ((np.arange(-1, n_bins + 1)) / n_bins_per_octave)  # center frequencies
    # centers in bins
    kc = np.round(f_cq * (n_fft / sr)).astype(int)
    c_mat = np.zeros([n_bins, int(np.round(n_fft / 2))])
    for k in range(1, kc.shape[0] - 1):
        l1 = kc[k] - kc[k - 1]
        w1 = scipy.signal.triang((l1 * 2) + 1)
        l2 = kc[k + 1] - kc[k]
        w2 = scipy.signal.triang((l2 * 2) + 1)
        wk = np.hstack(
            [w1[0:l1], w2[l2:]])  # concatenate two halves. l1 and l2 are different because of the log-spacing
        c_mat[k - 1, kc[k - 1]:(kc[k + 1] + 1)] = wk / np.sum(wk)  # normalized to unit sum;
    return c_mat, f_cq  # matrix with triangular filterbank


def onset_detection_fn(x, n_fft, win_length, hop_length, n_bins_per_octave, n_octaves, f_min, sr, mean_filter_size):
    """
    Filter bank for onset pattern calculation
    """
    # calculate frequency constant-q transform
    f_win = scipy.signal.hann(win_length)
    x_spec = librosa.stft(x,
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length,
                          window=f_win)
    x_spec = np.abs(x_spec) / (2 * np.sum(f_win))

    # get CQ Transform 
    f_cq_mat, f_cq = cq_matrix(n_bins_per_octave, n_octaves * n_bins_per_octave, f_min, win_length, sr)
    x_cq_spec = np.dot(f_cq_mat, x_spec[:-1, :])

    # subtract moving mean
    # DIFFERENCE BETWEEN THE CURRENT FRAME AND THE MEAN OF THE PREVIOUS mean_filter_size FRAMES 
    b = np.concatenate([[1], np.ones(mean_filter_size, dtype=float) / -mean_filter_size])
    od_fun = scipy.signal.lfilter(b, 1, x_cq_spec, axis=1)

    # half-wave rectify
    od_fun = np.maximum(0, od_fun)

    # post-process OPs
    od_fun = np.log10(1 + 1000 * od_fun)  ## log scaling
    od_fun = np.abs(od_fun).astype('float32')
    od_fun = np.moveaxis(od_fun, 1, 0)
    # clip
    od_fun = np.clip(od_fun / 2.25, 0, 1)  # 2.25 ?????????

    # get logf_stft
    logf_stft = librosa.power_to_db(x_cq_spec).astype('float32')
    logf_stft = np.moveaxis(logf_stft, 1, 0)

    return od_fun, logf_stft, f_cq


def reduce_frequency_bands_in_spectrogram(freq_out, freq_in, S):
    """
    @param freq_out:        band center frequencies in output spectrogram
    @param freq_in:         band center frequencies in input spectrogram
    @param S:               spectrogram to reduce
    @returns S_out:         spectrogram reduced in frequency
    """

    if len(freq_out) >= len(freq_in):
        warnings.warn(
            "Number of bands in reduced spectrogram should be smaller than initial number of bands in spectrogram")

    n_timeframes = S.shape[0]
    n_bands = len(freq_out)

    # find index of closest input frequency
    freq_out_idx = np.array([], dtype=int)

    for f in freq_out:
        freq_out_idx = np.append(freq_out_idx, np.abs(freq_in - f).argmin())

    # band limits (not center)
    freq_out_band_idx = np.array([0], dtype=int)

    for i in range(len(freq_out_idx) - 1):
        li = np.ceil((freq_out_idx[i + 1] - freq_out_idx[i]) / 2) + freq_out_idx[i]  # find left border of band
        freq_out_band_idx = np.append(freq_out_band_idx, [li])

    freq_out_band_idx = np.append(freq_out_band_idx, len(freq_in))  # add last frequency in input spectrogram
    freq_out_band_idx = np.array(freq_out_band_idx, dtype=int)  # convert to int

    # init empty spectrogram
    S_out = np.zeros([n_timeframes, n_bands])

    # reduce spectrogram
    for i in range(len(freq_out_band_idx) - 1):
        li = freq_out_band_idx[i] + 1  # band left index
        if i == 0: li = 0
        ri = freq_out_band_idx[i + 1]  # band right index
        S_out[:, i] = np.max(S[:, li:ri], axis=1)  # pooling

    return S_out


def get_grid_timestamps(n_bars=2, time_signature_numerator=4, time_signature_denominator=4,
                        beat_division_factors=[4], qpm=120):
    """
    @param n_bars                               Number of bars
    @param time_signature_numerator             Time signature numerator
    @param time_signature_denominator           Time signature denominator
    @param beat_division_factors                Array of beat division factors
    @param qpm                                  Quarter per minute
    @return grid                                Array of timestamps of grid lines
    
    """

    hvo_seq = HVO_Sequence()
    hvo_seq.add_time_signature(time_step=0, numerator=time_signature_numerator, denominator=time_signature_denominator,
                               beat_division_factors=beat_division_factors)
    hvo_seq.add_tempo(time_step=0, qpm=qpm)

    # total number of steps is total_number_of_bars*time_signature_numerator*beat_division_factor
    n_timesteps = 0
    for factor in beat_division_factors:
        n_timesteps += n_bars * time_signature_numerator * factor

    # Create a zero hvo
    hits = np.zeros([n_timesteps, 1])
    vels = np.zeros([n_timesteps, 1])
    offs = np.zeros([n_timesteps, 1])

    # Add hvo score to hvo_seq instance
    hvo_seq.hvo = np.concatenate((hits, vels, offs), axis=1)

    # get grid lines
    # hvo_seq.grid_lines_with_types
    grid = hvo_seq.grid_lines
    # grid_len = hvo_seq.total_len

    return grid


def get_onset_detect(onset_strength):
    """
    
    """
    n_timeframes = onset_strength.shape[0]
    n_bands = onset_strength.shape[1]

    onset_detect = np.zeros([n_timeframes, n_bands])

    for band in range(n_bands):
        time_frame_idx = librosa.onset.onset_detect(onset_envelope=onset_strength.T[band, :])
        onset_detect[time_frame_idx, band] = 1

    return onset_detect


def map_onsets_to_grid(grid, onset_strength, onset_detect, hop_length, n_fft, sr):
    """
    Maps matrices of onset strength and onset detection into a grid with a lower temporal resolution.
    @param grid:                 Array with timestamps
    @param onset_strength:       Matrix of onset strength values (n_timeframes x n_bands)
    @param onset_detect:         Matrix of onset detection (1,0) (n_timeframes x n_bands)
    @param hop_length:
    @param n_fft
    @return onsets_grid:         Onsets with respect to lines in grid (len_grid x n_bands)
    @return intensity_grid:      Strength values for each detected onset (len_grid x n_bands)
    """

    if onset_strength.shape != onset_detect.shape:
        warnings.warn(
            f"onset_strength shape and onset_detect shape must be equal. Instead, got {onset_strength.shape} and {onset_detect.shape}")

    n_bands = onset_strength.shape[1]
    n_timeframes = onset_detect.shape[0]
    n_timesteps = len(grid) - 1 # last grid line is first line of next bar

    # init intensity and onsets grid
    strength_grid = np.zeros([n_timesteps, n_bands])
    onsets_grid = np.zeros([n_timesteps, n_bands])

    # time array
    time = librosa.frames_to_time(np.arange(n_timeframes), sr=sr,
                                  hop_length=hop_length, n_fft=n_fft)

    # map onsets and strength into grid
    for band in range(n_bands):
        for timeframe_idx in range(n_timeframes):
            if onset_detect[timeframe_idx, band]:  # if there is an onset detected, get grid index and utiming
                grid_idx, utiming = get_grid_position_and_utiming_in_hvo(time[timeframe_idx], grid)
                if grid_idx == n_timesteps : continue # in case that a hit is assigned to last grid line
                strength_grid[grid_idx, band] = onset_strength[timeframe_idx, band]
                onsets_grid[grid_idx, band] = utiming

    return onsets_grid, strength_grid


def input_features_extractor(audio_file_path=None, **kwargs):
    # default values
    sr = kwargs.get('sr', 44100)
    n_fft = kwargs.get('n_fft', 1024)
    win_length = kwargs.get('win_length', 1024)
    hop_length = kwargs.get('hop_length', 512)
    n_bins_per_octave = kwargs.get('n_bins_per_octave', 16)
    n_octaves = kwargs.get('n_octaves', 9)
    f_min = kwargs.get('f_min', 40)
    mean_filter_size = kwargs.get('mean_filter_size', 22)
    c_freq = kwargs.get('c_freq', [55, 90, 138, 175, 350, 6000, 8500, 12500])
    n_bars = kwargs.get('n_bars', 2)
    time_signature_numerator = kwargs.get('time_signature_numerator', 4)
    time_signature_denominator = kwargs.get('time_signature_denominator', 4)
    beat_division_factors = kwargs.get('beat_division_factors', [4])
    qpm = kwargs.get('qpm', 120)

    x, sr = read_audio(audio_file_path, mono=True, sr=sr)
    x /= np.max(np.abs(x))

    mb_onset_strength, logf_stft, f_cq = onset_detection_fn(x,
                                                            n_fft,
                                                            win_length,
                                                            hop_length,
                                                            n_bins_per_octave,
                                                            n_octaves,
                                                            f_min,
                                                            sr,
                                                            mean_filter_size)

    mb_onset_strength = reduce_frequency_bands_in_spectrogram(c_freq, f_cq, mb_onset_strength)
    mb_onset_detect = get_onset_detect(mb_onset_strength)

    grid = get_grid_timestamps(n_bars=n_bars,
                               time_signature_numerator=time_signature_numerator,
                               time_signature_denominator=time_signature_denominator,
                               beat_division_factors=beat_division_factors,
                               qpm=qpm)
    onsets_grid, strength_grid = map_onsets_to_grid(grid, mb_onset_strength, mb_onset_detect, n_fft=n_fft,
                                                    hop_length=hop_length, sr=sr)

    input_features = np.concatenate((onsets_grid,strength_grid), axis=1)

    return input_features
