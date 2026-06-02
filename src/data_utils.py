"""
Dataset utilities for ASVspoof with parameterized sample rate.

This module provides dataset classes that support different sample rates
for wav2vec2 (16kHz) and MiMo (24kHz) frontends.

Based on SSL_Anti-spoofing by Hemlata Tak.
"""

from typing import Dict, List, Optional, Tuple, Any

import librosa
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .rawboost import (
    ISD_additive_noise,
    LnL_convolutive_noise,
    SSI_additive_noise,
    normWav,
    scale_rawboost_params,
    REFERENCE_SAMPLE_RATE,
)


def genSpoof_list(
    dir_meta: str, is_train: bool = False, is_eval: bool = False
) -> Tuple[Dict[str, int], List[str]] | List[str]:
    """
    Generate file list and labels from metadata file.

    Args:
        dir_meta: Path to metadata file
        is_train: Whether this is training data
        is_eval: Whether this is evaluation data (no labels)

    Returns:
        For train/dev: (label_dict, file_list)
        For eval: file_list only
    """
    d_meta = {}
    file_list = []

    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split()
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split()
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def pad(x: np.ndarray, max_len: int = 64600) -> np.ndarray:
    """
    Pad or truncate audio to fixed length.

    Args:
        x: Audio waveform
        max_len: Target length in samples

    Returns:
        Padded/truncated waveform
    """
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # Need to pad by repeating
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def get_cut_length(sample_rate: int, duration_sec: float = 4.0) -> int:
    """
    Calculate cut length for a given sample rate and duration.

    Args:
        sample_rate: Sample rate in Hz
        duration_sec: Target duration in seconds

    Returns:
        Number of samples
    """
    # Baseline uses 64600 samples at 16kHz (~4.04s)
    # For consistency, we use the same duration ratio
    if sample_rate == 16000:
        return 64600
    elif sample_rate == 24000:
        # 64600 / 16000 * 24000 = 96900, round to 96000
        return 96000
    else:
        return int(sample_rate * duration_sec)


class Dataset_ASVspoof2019_train(Dataset):
    """
    Training dataset for ASVspoof2019 with RawBoost augmentation.

    Args:
        args: RawBoost configuration arguments
        list_IDs: List of utterance keys
        labels: Dictionary mapping keys to labels (0=spoof, 1=bonafide)
        base_dir: Base directory containing audio files
        algo: RawBoost algorithm (0=none, 1-8=various augmentations)
        sample_rate: Target sample rate (16000 for wav2vec2, 24000 for MiMo)
        cut: Number of samples to use (None = auto based on sample_rate)
        auto_scale_rawboost: If True, automatically scale RawBoost params for
                             sample rates different from 16kHz reference.
                             Default: True for MiMo compatibility.
    """

    def __init__(
        self,
        args: Any,
        list_IDs: List[str],
        labels: Dict[str, int],
        base_dir: str,
        algo: int,
        sample_rate: int = 16000,
        cut: Optional[int] = None,
        auto_scale_rawboost: bool = True,
    ):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo = algo
        self.sample_rate = sample_rate
        self.cut = cut if cut is not None else get_cut_length(sample_rate)
        self.auto_scale_rawboost = auto_scale_rawboost

        # Scale RawBoost parameters if needed
        if auto_scale_rawboost and sample_rate != REFERENCE_SAMPLE_RATE:
            self.args = scale_rawboost_params(args, sample_rate, auto_scale=True)
        else:
            self.args = args

    def __len__(self) -> int:
        return len(self.list_IDs)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(
            self.base_dir + "flac/" + utt_id + ".flac", sr=self.sample_rate
        )
        Y = process_Rawboost_feature(X, self.sample_rate, self.args, self.algo)
        X_pad = pad(Y, self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id]

        return x_inp, target


class Dataset_ASVspoof2021_eval(Dataset):
    """
    Evaluation dataset for ASVspoof2021.

    Args:
        list_IDs: List of utterance keys
        base_dir: Base directory containing audio files
        sample_rate: Target sample rate (16000 for wav2vec2, 24000 for MiMo)
        cut: Number of samples to use (None = auto based on sample_rate)
    """

    def __init__(
        self,
        list_IDs: List[str],
        base_dir: str,
        sample_rate: int = 16000,
        cut: Optional[int] = None,
    ):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.sample_rate = sample_rate
        self.cut = cut if cut is not None else get_cut_length(sample_rate)

    def __len__(self) -> int:
        return len(self.list_IDs)

    def __getitem__(self, index: int) -> Tuple[Tensor, str]:
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(
            self.base_dir + "flac/" + utt_id + ".flac", sr=self.sample_rate
        )
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id


class Dataset_ASVspoof2021_fast_eval(Dataset):
    """
    Fast evaluation dataset for ASVspoof2021 with labels.

    Returns (audio, label, file_id) for loss and EER computation during training.

    Args:
        list_IDs: List of utterance keys
        labels: Dictionary mapping file_id to label ('bonafide' or 'spoof')
        base_dir: Base directory containing audio files
        sample_rate: Target sample rate (16000 for wav2vec2, 24000 for MiMo)
        cut: Number of samples to use (None = auto based on sample_rate)
    """

    def __init__(
        self,
        list_IDs: List[str],
        labels: Dict[str, str],
        base_dir: str,
        sample_rate: int = 16000,
        cut: Optional[int] = None,
    ):
        self.list_IDs = list_IDs
        self.labels = labels  # file_id -> 'bonafide' or 'spoof'
        self.base_dir = base_dir
        self.sample_rate = sample_rate
        self.cut = cut if cut is not None else get_cut_length(sample_rate)

    def __len__(self) -> int:
        return len(self.list_IDs)

    def __getitem__(self, index: int) -> Tuple[Tensor, int, str]:
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(
            self.base_dir + "flac/" + utt_id + ".flac", sr=self.sample_rate
        )
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        # Convert label to int: 1=bonafide, 0=spoof
        label = 1 if self.labels.get(utt_id, 'spoof') == 'bonafide' else 0
        return x_inp, label, utt_id


# --------------------- RawBoost Processing --------------------- #


def process_Rawboost_feature(
    feature: np.ndarray, sr: int, args: Any, algo: int
) -> np.ndarray:
    """
    Apply RawBoost augmentation to audio feature.

    Args:
        feature: Audio waveform
        sr: Sample rate
        args: RawBoost configuration (contains noise parameters).
              Can be either a config object or ScaledRawBoostParams.
              If ScaledRawBoostParams, frequency-dependent params are already scaled.
        algo: Algorithm selection:
            0: No augmentation
            1: Convolutive noise
            2: Impulsive noise
            3: Coloured additive noise
            4: All three in series (1+2+3)
            5: Convolutive + Impulsive (1+2)
            6: Convolutive + Coloured (1+3)
            7: Impulsive + Coloured (2+3)
            8: Convolutive || Impulsive in parallel (1||2)

    Returns:
        Augmented waveform
    """
    # Get maxF - if using ScaledRawBoostParams, it's already scaled
    # Otherwise, cap at Nyquist limit for safety
    maxF = getattr(args, "maxF", sr // 2 - 100)
    nyquist = sr // 2
    if maxF > nyquist:
        maxF = nyquist - 100

    # Get other params (works for both config objects and dataclasses)
    minBW = getattr(args, "minBW", 100)
    maxBW = getattr(args, "maxBW", 1000)
    minCoeff = getattr(args, "minCoeff", 10)
    maxCoeff = getattr(args, "maxCoeff", 100)

    # Get remaining params (not scaled)
    N_f = getattr(args, "N_f", 5)
    nBands = getattr(args, "nBands", 5)
    minF = getattr(args, "minF", 20)
    minG = getattr(args, "minG", 0)
    maxG = getattr(args, "maxG", 0)
    minBiasLinNonLin = getattr(args, "minBiasLinNonLin", 5)
    maxBiasLinNonLin = getattr(args, "maxBiasLinNonLin", 20)
    P = getattr(args, "P", 10)
    g_sd = getattr(args, "g_sd", 2)
    SNRmin = getattr(args, "SNRmin", 10)
    SNRmax = getattr(args, "SNRmax", 40)

    # Data process by Convolutive noise (1st algo)
    if algo == 1:
        feature = LnL_convolutive_noise(
            feature, N_f, nBands, minF, maxF, minBW, maxBW,
            minCoeff, maxCoeff, minG, maxG, minBiasLinNonLin, maxBiasLinNonLin, sr,
        )

    # Data process by Impulsive noise (2nd algo)
    elif algo == 2:
        feature = ISD_additive_noise(feature, P, g_sd)

    # Data process by coloured additive noise (3rd algo)
    elif algo == 3:
        feature = SSI_additive_noise(
            feature, SNRmin, SNRmax, nBands, minF, maxF,
            minBW, maxBW, minCoeff, maxCoeff, minG, maxG, sr,
        )

    # Data process by all 3 algo. together in series (1+2+3)
    elif algo == 4:
        feature = LnL_convolutive_noise(
            feature, N_f, nBands, minF, maxF, minBW, maxBW,
            minCoeff, maxCoeff, minG, maxG, minBiasLinNonLin, maxBiasLinNonLin, sr,
        )
        feature = ISD_additive_noise(feature, P, g_sd)
        feature = SSI_additive_noise(
            feature, SNRmin, SNRmax, nBands, minF, maxF,
            minBW, maxBW, minCoeff, maxCoeff, minG, maxG, sr,
        )

    # Data process by 1st two algo. together in series (1+2)
    elif algo == 5:
        feature = LnL_convolutive_noise(
            feature, N_f, nBands, minF, maxF, minBW, maxBW,
            minCoeff, maxCoeff, minG, maxG, minBiasLinNonLin, maxBiasLinNonLin, sr,
        )
        feature = ISD_additive_noise(feature, P, g_sd)

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo == 6:
        feature = LnL_convolutive_noise(
            feature, N_f, nBands, minF, maxF, minBW, maxBW,
            minCoeff, maxCoeff, minG, maxG, minBiasLinNonLin, maxBiasLinNonLin, sr,
        )
        feature = SSI_additive_noise(
            feature, SNRmin, SNRmax, nBands, minF, maxF,
            minBW, maxBW, minCoeff, maxCoeff, minG, maxG, sr,
        )

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo == 7:
        feature = ISD_additive_noise(feature, P, g_sd)
        feature = SSI_additive_noise(
            feature, SNRmin, SNRmax, nBands, minF, maxF,
            minBW, maxBW, minCoeff, maxCoeff, minG, maxG, sr,
        )

    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo == 8:
        feature1 = LnL_convolutive_noise(
            feature, N_f, nBands, minF, maxF, minBW, maxBW,
            minCoeff, maxCoeff, minG, maxG, minBiasLinNonLin, maxBiasLinNonLin, sr,
        )
        feature2 = ISD_additive_noise(feature, P, g_sd)
        feature_para = feature1 + feature2
        feature = normWav(feature_para, 0)  # normalized resultant waveform

    # original data without Rawboost processing (algo == 0 or unknown)
    # feature remains unchanged

    return feature
