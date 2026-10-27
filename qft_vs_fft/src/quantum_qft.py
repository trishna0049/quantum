"""Manual Quantum Fourier Transform circuit construction and simulation."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def qft(n: int) -> QuantumCircuit:
    """Build the n-qubit QFT using Hadamard, controlled-phase, and swap gates.

    With Qiskit's little-endian statevector basis ordering, this circuit uses
    the positive-exponent convention:
    QFT|x> = sum_y exp(2*pi*i*x*y/N)|y> / sqrt(N).
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")

    qc = QuantumCircuit(n, name="QFT")
    for j in reversed(range(n)):
        qc.h(j)
        for k in reversed(range(j)):
            qc.cp(np.pi / 2**(j - k), k, j)
    for i in range(n // 2):
        qc.swap(i, n - i - 1)
    return qc


def inverse_qft(n: int) -> QuantumCircuit:
    """Build the inverse of the manually constructed n-qubit QFT."""
    return qft(n).inverse()


def qft_statevector(values: np.ndarray) -> np.ndarray:
    """Apply the QFT to a normalized input vector and return its amplitudes.

    An arbitrary vector must be normalized before it can represent a quantum
    state. The returned state therefore corresponds to values / ||values||.
    """
    amplitudes = np.asarray(values, dtype=complex)
    if amplitudes.ndim != 1 or amplitudes.size == 0:
        raise ValueError("values must be a non-empty one-dimensional vector")
    n_float = np.log2(amplitudes.size)
    if not n_float.is_integer():
        raise ValueError("values length must be a power of two")
    norm = np.linalg.norm(amplitudes)
    if np.isclose(norm, 0.0):
        raise ValueError("values must have a non-zero norm")

    normalized = amplitudes / norm
    circuit = qft(int(n_float))
    return Statevector(normalized).evolve(circuit).data
