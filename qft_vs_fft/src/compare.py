"""Correctness, complexity, depth, and runtime comparison helpers."""

from time import perf_counter

import numpy as np
from qiskit.quantum_info import Statevector

from src.classical_fft import numpy_qft_reference
from src.quantum_qft import qft


def normalize(values: np.ndarray) -> np.ndarray:
    """Normalize a nonzero, one-dimensional vector for quantum state use."""
    vector = np.asarray(values, dtype=complex)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("values must be a non-empty one-dimensional vector")
    norm = np.linalg.norm(vector)
    if np.isclose(norm, 0.0):
        raise ValueError("values must have a non-zero norm")
    return vector / norm


def comparison_cases(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Build representative inputs of length 2**n."""
    if n < 1:
        raise ValueError("n must be a positive integer")
    size = 2**n
    indices = np.arange(size)
    frequency = max(1, size // 4)
    return {
        "random complex": rng.normal(size=size) + 1j * rng.normal(size=size),
        "basis state": np.eye(1, size, size // 3, dtype=complex)[0],
        "uniform": np.ones(size, dtype=complex),
        "periodic cosine": np.cos(2 * np.pi * frequency * indices / size),
    }


def compare_input(values: np.ndarray, atol: float = 1e-10) -> tuple[bool, float]:
    """Compare Qiskit's simulated QFT with the matching classical reference."""
    normalized = normalize(values)
    quantum = Statevector(normalized).evolve(qft(int(np.log2(normalized.size)))).data
    classical = numpy_qft_reference(normalized)
    error = float(np.max(np.abs(quantum - classical)))
    return bool(np.allclose(quantum, classical, atol=atol)), error


def correctness_results(
    min_qubits: int = 1, max_qubits: int = 5, seed: int = 42
) -> list[dict[str, int | str | bool | float]]:
    """Compare four input families for every requested qubit count."""
    if min_qubits < 1 or max_qubits < min_qubits:
        raise ValueError("require 1 <= min_qubits <= max_qubits")

    rng = np.random.default_rng(seed)
    results = []
    for n in range(min_qubits, max_qubits + 1):
        for label, values in comparison_cases(n, rng).items():
            match, error = compare_input(values)
            results.append(
                {"n": n, "N": 2**n, "input": label, "match": match, "max_error": error}
            )
    return results


def qft_gate_count(n: int, verify_with_circuit: bool = True) -> int:
    """Count H, CP, and SWAP gates, checking actual circuits for small n."""
    if n < 1:
        raise ValueError("n must be a positive integer")
    if verify_with_circuit and n <= 8:
        return int(sum(qft(n).count_ops().values()))
    return n * (n + 1) // 2 + n // 2


def qft_depth(n: int) -> int:
    """Return the transpiler-free depth of the manually built QFT circuit."""
    if n < 1:
        raise ValueError("n must be a positive integer")
    return int(qft(n).depth())


def timing_results(
    min_qubits: int = 1, max_qubits: int = 14, repeats: int = 3, seed: int = 7
) -> list[dict[str, int | float]]:
    """Measure NumPy FFT and local statevector QFT simulation runtimes.

    The statevector timing includes constructing the input Statevector and
    evolving it through the QFT circuit, but excludes circuit construction.
    """
    if min_qubits < 1 or max_qubits < min_qubits or repeats < 1:
        raise ValueError("require 1 <= min_qubits <= max_qubits and repeats >= 1")
    rng = np.random.default_rng(seed)
    results = []

    for n in range(min_qubits, max_qubits + 1):
        size = 2**n
        values = normalize(rng.normal(size=size) + 1j * rng.normal(size=size))
        circuit = qft(n)
        fft_times = []
        simulation_times = []

        for _ in range(repeats):
            start = perf_counter()
            np.fft.fft(values)
            fft_times.append(perf_counter() - start)

            start = perf_counter()
            Statevector(values).evolve(circuit)
            simulation_times.append(perf_counter() - start)

        results.append(
            {
                "n": n,
                "N": size,
                "numpy_fft_seconds": float(np.median(fft_times)),
                "statevector_seconds": float(np.median(simulation_times)),
            }
        )
    return results
