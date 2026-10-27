"""Correctness tests for the manually constructed Quantum Fourier Transform."""

import numpy as np
from qiskit.quantum_info import Operator, Statevector

from src.classical_fft import numpy_qft_reference
from src.quantum_qft import inverse_qft, qft


def test_qft_matches_positive_exponent_reference_for_random_inputs() -> None:
    rng = np.random.default_rng(2025)
    for n in range(1, 6):
        size = 2**n
        values = rng.normal(size=size) + 1j * rng.normal(size=size)
        normalized = values / np.linalg.norm(values)
        actual = Statevector(normalized).evolve(qft(n)).data
        expected = np.fft.ifft(normalized) * np.sqrt(size)
        assert np.allclose(actual, expected, atol=1e-10)
        assert np.allclose(actual, numpy_qft_reference(normalized), atol=1e-10)


def test_qft_of_zero_basis_state_is_uniform_superposition() -> None:
    for n in range(1, 6):
        state = Statevector.from_int(0, 2**n).evolve(qft(n)).data
        expected = np.full(2**n, 1 / np.sqrt(2**n), dtype=complex)
        assert np.allclose(state, expected, atol=1e-10)


def test_inverse_qft_composed_with_qft_is_identity() -> None:
    for n in range(1, 6):
        combined = qft(n).compose(inverse_qft(n))
        assert np.allclose(Operator(combined).data, np.eye(2**n), atol=1e-10)


def test_qft_operator_is_unitary() -> None:
    for n in range(1, 6):
        matrix = Operator(qft(n)).data
        identity = np.eye(2**n)
        assert np.allclose(matrix.conj().T @ matrix, identity, atol=1e-10)
