"""Classical discrete Fourier transform implementations."""

import numpy as np


def _validate_signal(values: np.ndarray) -> np.ndarray:
    signal = np.asarray(values, dtype=complex)
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("values must be a non-empty one-dimensional vector")
    return signal


def naive_dft(values: np.ndarray) -> np.ndarray:
    """Compute the positive-exponent DFT directly in O(N^2) operations.

    The positive sign matches the QFT convention. NumPy's inverse FFT,
    multiplied by sqrt(N), is an efficient reference for this same
    unnormalized sum.
    """
    signal = _validate_signal(values)
    size = signal.size
    indices = np.arange(size)
    kernel = np.exp(2j * np.pi * np.outer(indices, indices) / size)
    return kernel @ signal / np.sqrt(size)


def numpy_fft(values: np.ndarray) -> np.ndarray:
    """Return NumPy's conventional negative-exponent forward FFT."""
    return np.fft.fft(_validate_signal(values))


def numpy_qft_reference(values: np.ndarray) -> np.ndarray:
    """Return the classical positive-exponent QFT convention for values.

    QFT|x> = np.fft.ifft(x) * sqrt(N), not np.fft.fft(x). For quantum use,
    normalize values first; this function itself preserves the input scale.
    """
    signal = _validate_signal(values)
    return np.fft.ifft(signal) * np.sqrt(signal.size)
