"""
RawBoost: Raw Data Boosting and Augmentation for Anti-Spoofing.

Based on:
Hemlata Tak, Madhu Kamble, Jose Patino, Massimiliano Todisco, Nicholas Evans.
RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic
Speaker Verification Anti-Spoofing. In Proc. ICASSP 2022, pp:6382--6386.

Extended with sample-rate aware parameter scaling for MiMo (24kHz) support.
"""

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal


# Reference sample rate (baseline parameters tuned for this)
REFERENCE_SAMPLE_RATE = 16000


@dataclass
class ScaledRawBoostParams:
    """
    RawBoost parameters with sample-rate aware scaling.

    The baseline RawBoost parameters were tuned for 16kHz audio.
    For higher sample rates (e.g., 24kHz for MiMo), frequency-dependent
    parameters need scaling to maintain similar perceptual effects.

    Scaled parameters:
    - maxF: Extended to use full spectrum (capped at Nyquist - 100Hz)
    - minBW, maxBW: Scaled proportionally
    - minCoeff, maxCoeff: Scaled for same frequency resolution

    Unchanged parameters:
    - N_f, nBands, minF: Not frequency-dependent or low enough
    - minG, maxG, minBiasLinNonLin, maxBiasLinNonLin: Amplitude-based
    - P, g_sd: Percentage-based
    - SNRmin, SNRmax: dB-based
    """
    # Scaled parameters
    maxF: float
    minBW: float
    maxBW: float
    minCoeff: int
    maxCoeff: int

    # Passed through unchanged
    N_f: int
    nBands: int
    minF: float
    minG: float
    maxG: float
    minBiasLinNonLin: float
    maxBiasLinNonLin: float
    P: float
    g_sd: float
    SNRmin: float
    SNRmax: float

    # Scaling metadata
    sample_rate: int
    scale_factor: float


def scale_rawboost_params(args: Any, sample_rate: int, auto_scale: bool = True) -> ScaledRawBoostParams:
    """
    Scale RawBoost parameters for the target sample rate.

    Args:
        args: RawBoost configuration object with baseline parameters.
              May contain optional 'bw_scale' for custom bandwidth scaling.
        sample_rate: Target sample rate (e.g., 16000 or 24000)
        auto_scale: If True, automatically scale frequency-dependent params.
                    If False, use params as-is (for manual override).

    Returns:
        ScaledRawBoostParams with appropriately scaled values

    Example:
        # For 24kHz (MiMo), with 16kHz baseline params:
        # scale_factor = 24000 / 16000 = 1.5
        # maxF: 8000 -> min(8000 * 1.5, 12000 - 100) = 11900
        # minBW: 100 -> 150
        # maxBW: 1000 -> 1500
        # minCoeff: 10 -> 15
        # maxCoeff: 100 -> 150
    """
    scale_factor = sample_rate / REFERENCE_SAMPLE_RATE if auto_scale else 1.0
    nyquist = sample_rate // 2

    # Get base values
    base_maxF = getattr(args, "maxF", 8000)
    base_minBW = getattr(args, "minBW", 100)
    base_maxBW = getattr(args, "maxBW", 1000)
    base_minCoeff = getattr(args, "minCoeff", 10)
    base_maxCoeff = getattr(args, "maxCoeff", 100)

    # Check for custom bandwidth scaling (for Optuna tuning)
    custom_bw_scale = getattr(args, "bw_scale", None)

    # Scale frequency-dependent parameters
    if auto_scale and scale_factor != 1.0:
        scaled_maxF = min(base_maxF * scale_factor, nyquist - 100)

        # Use custom bw_scale if provided, otherwise use scale_factor
        bw_multiplier = custom_bw_scale if custom_bw_scale is not None else scale_factor
        scaled_minBW = base_minBW * bw_multiplier
        scaled_maxBW = base_maxBW * bw_multiplier

        scaled_minCoeff = int(base_minCoeff * scale_factor)
        scaled_maxCoeff = int(base_maxCoeff * scale_factor)
    else:
        # Use values as-is, but still cap maxF at Nyquist
        scaled_maxF = min(base_maxF, nyquist - 100)

        # Still apply custom bw_scale if provided
        if custom_bw_scale is not None:
            scaled_minBW = base_minBW * custom_bw_scale
            scaled_maxBW = base_maxBW * custom_bw_scale
        else:
            scaled_minBW = base_minBW
            scaled_maxBW = base_maxBW

        scaled_minCoeff = base_minCoeff
        scaled_maxCoeff = base_maxCoeff

    return ScaledRawBoostParams(
        maxF=scaled_maxF,
        minBW=scaled_minBW,
        maxBW=scaled_maxBW,
        minCoeff=scaled_minCoeff,
        maxCoeff=scaled_maxCoeff,
        # Pass through unchanged
        N_f=getattr(args, "N_f", 5),
        nBands=getattr(args, "nBands", 5),
        minF=getattr(args, "minF", 20),
        minG=getattr(args, "minG", 0),
        maxG=getattr(args, "maxG", 0),
        minBiasLinNonLin=getattr(args, "minBiasLinNonLin", 5),
        maxBiasLinNonLin=getattr(args, "maxBiasLinNonLin", 20),
        P=getattr(args, "P", 10),
        g_sd=getattr(args, "g_sd", 2),
        SNRmin=getattr(args, "SNRmin", 10),
        SNRmax=getattr(args, "SNRmax", 40),
        sample_rate=sample_rate,
        scale_factor=scale_factor,
    )


def randRange(x1: float, x2: float, integer: bool) -> float:
    """Generate random value in range."""
    y = np.random.uniform(low=x1, high=x2)
    if integer:
        y = int(y)
    return y


def normWav(x: np.ndarray, always: bool) -> np.ndarray:
    """Normalize waveform."""
    if always:
        x = x / np.amax(abs(x))
    elif np.amax(abs(x)) > 1:
        x = x / np.amax(abs(x))
    return x


def genNotchCoeffs(
    nBands: int,
    minF: float,
    maxF: float,
    minBW: float,
    maxBW: float,
    minCoeff: int,
    maxCoeff: int,
    minG: float,
    maxG: float,
    fs: int,
) -> np.ndarray:
    """Generate notch filter coefficients."""
    b = 1
    for i in range(0, nBands):
        fc = randRange(minF, maxF, 0)
        bw = randRange(minBW, maxBW, 0)
        c = randRange(minCoeff, maxCoeff, 1)

        if c / 2 == int(c / 2):
            c = c + 1
        f1 = fc - bw / 2
        f2 = fc + bw / 2
        if f1 <= 0:
            f1 = 1 / 1000
        if f2 >= fs / 2:
            f2 = fs / 2 - 1 / 1000
        b = np.convolve(
            signal.firwin(c, [float(f1), float(f2)], window="hamming", fs=fs), b
        )

    G = randRange(minG, maxG, 0)
    _, h = signal.freqz(b, 1, fs=fs)
    b = pow(10, G / 20) * b / np.amax(abs(h))
    return b


def filterFIR(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Apply FIR filter."""
    N = b.shape[0] + 1
    xpad = np.pad(x, (0, N), "constant")
    y = signal.lfilter(b, 1, xpad)
    y = y[int(N / 2) : int(y.shape[0] - N / 2)]
    return y


def LnL_convolutive_noise(
    x: np.ndarray,
    N_f: int,
    nBands: int,
    minF: float,
    maxF: float,
    minBW: float,
    maxBW: float,
    minCoeff: int,
    maxCoeff: int,
    minG: float,
    maxG: float,
    minBiasLinNonLin: float,
    maxBiasLinNonLin: float,
    fs: int,
) -> np.ndarray:
    """Linear and non-linear convolutive noise."""
    y = [0] * x.shape[0]
    for i in range(0, N_f):
        if i == 1:
            minG = minG - minBiasLinNonLin
            maxG = maxG - maxBiasLinNonLin
        b = genNotchCoeffs(
            nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs
        )
        y = y + filterFIR(np.power(x, (i + 1)), b)
    y = y - np.mean(y)
    y = normWav(y, 0)
    return y


def ISD_additive_noise(x: np.ndarray, P: float, g_sd: float) -> np.ndarray:
    """Impulsive signal dependent noise."""
    beta = randRange(0, P, 0)

    y = copy.deepcopy(x)
    x_len = x.shape[0]
    n = int(x_len * (beta / 100))
    p = np.random.permutation(x_len)[:n]
    f_r = np.multiply(
        ((2 * np.random.rand(p.shape[0])) - 1), ((2 * np.random.rand(p.shape[0])) - 1)
    )
    r = g_sd * x[p] * f_r
    y[p] = x[p] + r
    y = normWav(y, 0)
    return y


def SSI_additive_noise(
    x: np.ndarray,
    SNRmin: float,
    SNRmax: float,
    nBands: int,
    minF: float,
    maxF: float,
    minBW: float,
    maxBW: float,
    minCoeff: int,
    maxCoeff: int,
    minG: float,
    maxG: float,
    fs: int,
) -> np.ndarray:
    """Stationary signal independent noise."""
    noise = np.random.normal(0, 1, x.shape[0])
    b = genNotchCoeffs(
        nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs
    )
    noise = filterFIR(noise, b)
    noise = normWav(noise, 1)
    SNR = randRange(SNRmin, SNRmax, 0)
    noise = noise / np.linalg.norm(noise, 2) * np.linalg.norm(x, 2) / 10.0 ** (
        0.05 * SNR
    )
    x = x + noise
    return x
