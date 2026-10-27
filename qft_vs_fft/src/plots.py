"""Publication-ready figures for the QFT versus FFT project."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from src.classical_fft import numpy_qft_reference
from src.compare import qft_depth, qft_gate_count
from src.quantum_qft import qft


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _prepare_output(path: Path | str | None) -> Path:
    output = Path(path) if path is not None else DEFAULT_RESULTS_DIR
    output.mkdir(parents=True, exist_ok=True)
    return output


def save_circuit_diagrams(output_dir: Path | str | None = None) -> list[Path]:
    """Save circuit diagrams for three- and four-qubit QFT circuits."""
    output = _prepare_output(output_dir)
    paths = []
    for n in (3, 4):
        figure = qft(n).draw(output="mpl", fold=-1, scale=1.0)
        figure.suptitle(f"Manual {n}-Qubit Quantum Fourier Transform")
        path = output / f"qft_circuit_n{n}.png"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def plot_qft_vs_classical(
    values: np.ndarray, output_dir: Path | str | None = None
) -> Path:
    """Compare QFT and its positive-exponent classical reference."""
    output = _prepare_output(output_dir)
    signal = np.asarray(values, dtype=complex)
    if signal.ndim != 1 or signal.size == 0 or (signal.size & (signal.size - 1)):
        raise ValueError("values must be a non-empty vector with power-of-two length")
    norm = np.linalg.norm(signal)
    if np.isclose(norm, 0.0):
        raise ValueError("values must have a non-zero norm")
    normalized = signal / norm
    quantum = Statevector(normalized).evolve(qft(int(np.log2(signal.size)))).data
    classical = numpy_qft_reference(normalized)
    bins = np.arange(signal.size)

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(bins - 0.18, np.abs(quantum), width=0.36, label="QFT amplitudes")
    axes[0].bar(
        bins + 0.18,
        np.abs(classical),
        width=0.36,
        label="Classical reference amplitudes",
        alpha=0.8,
    )
    axes[0].set(title="Magnitude", xlabel="Frequency bin", ylabel="Magnitude")
    axes[0].legend()

    axes[1].bar(
        bins - 0.18, np.angle(quantum), width=0.36, label="QFT amplitudes"
    )
    axes[1].bar(
        bins + 0.18,
        np.angle(classical),
        width=0.36,
        label="Classical reference amplitudes",
        alpha=0.8,
    )
    axes[1].set(title="Phase", xlabel="Frequency bin", ylabel="Phase (radians)")
    axes[1].legend()
    figure.suptitle("QFT versus Classical Positive-Exponent Transform")
    figure.tight_layout()
    path = output / "qft_vs_classical_magnitude_phase.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_periodic_signal(
    output_dir: Path | str | None = None, n: int = 4
) -> Path:
    """Plot a cosine and its QFT magnitude spectrum with expected peak bins."""
    output = _prepare_output(output_dir)
    size = 2**n
    frequency = max(1, size // 4)
    indices = np.arange(size)
    signal = np.cos(2 * np.pi * frequency * indices / size)
    normalized = signal / np.linalg.norm(signal)
    spectrum = Statevector(normalized).evolve(qft(n)).data
    expected_bins = sorted({frequency % size, (-frequency) % size})

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].stem(indices, signal, basefmt=" ")
    axes[0].set(title=f"Periodic cosine (N={size})", xlabel="Sample index", ylabel="Amplitude")
    axes[1].stem(indices, np.abs(spectrum), basefmt=" ", label="QFT magnitude")
    axes[1].scatter(
        expected_bins,
        np.abs(spectrum)[expected_bins],
        color="red",
        zorder=3,
        label=f"Expected bins {expected_bins}",
    )
    axes[1].set(
        title="QFT spectrum",
        xlabel="Frequency bin",
        ylabel="Magnitude",
        xticks=indices,
    )
    axes[1].legend()
    figure.suptitle("Periodic Signal and Its Quantum Fourier Transform")
    figure.tight_layout()
    path = output / "periodic_signal_spectrum.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_complexity(output_dir: Path | str | None = None) -> Path:
    """Plot gate/operation growth for n=1..20 qubits (N=2**n)."""
    output = _prepare_output(output_dir)
    qubits = np.arange(1, 21)
    sizes = 2**qubits
    gates = np.array([qft_gate_count(int(n)) for n in qubits])
    classical_fft = sizes * qubits
    naive_dft = sizes**2

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(qubits, gates, marker="o", label="QFT gates (H + CP + SWAP)")
    axis.plot(qubits, classical_fft, marker="s", label="Classical FFT ~ N log2(N)")
    axis.plot(qubits, naive_dft, marker="^", label="Naive DFT ~ N^2")
    axis.set_yscale("log")
    axis.set(
        title="Asymptotic Operation Counts",
        xlabel="Number of qubits n (N = 2ⁿ)",
        ylabel="Gate / operation count (log scale)",
        xticks=np.arange(1, 21),
    )
    axis.grid(True, which="both", linestyle=":", alpha=0.5)
    axis.legend()
    figure.tight_layout()
    path = output / "complexity_comparison.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_circuit_depth(output_dir: Path | str | None = None) -> Path:
    """Plot the actual, non-transpiled QFT circuit depth for n=1..20."""
    output = _prepare_output(output_dir)
    qubits = np.arange(1, 21)
    depths = np.array([qft_depth(int(n)) for n in qubits])
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(qubits, depths, marker="o", color="tab:purple")
    axis.set(
        title="Manual QFT Circuit Depth",
        xlabel="Number of qubits n",
        ylabel="Circuit depth",
        xticks=qubits,
    )
    axis.grid(True, linestyle=":", alpha=0.6)
    figure.tight_layout()
    path = output / "circuit_depth.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_timing(
    output_dir: Path | str | None = None, max_qubits: int = 14
) -> tuple[Path, list[dict[str, int | float]]]:
    """Plot measured NumPy FFT and local statevector simulation runtimes."""
    from src.compare import timing_results

    output = _prepare_output(output_dir)
    timings = timing_results(max_qubits=max_qubits)
    qubits = [int(row["n"]) for row in timings]
    fft_seconds = [float(row["numpy_fft_seconds"]) for row in timings]
    simulation_seconds = [float(row["statevector_seconds"]) for row in timings]
    figure, axis = plt.subplots(figsize=(8.5, 5))
    axis.plot(qubits, fft_seconds, marker="o", label="NumPy FFT")
    axis.plot(qubits, simulation_seconds, marker="s", label="Qiskit Statevector QFT")
    axis.set_yscale("log")
    axis.set(
        title="Measured Runtime: FFT vs Classical Statevector Simulation",
        xlabel="Number of qubits n (input length N = 2ⁿ)",
        ylabel="Median runtime (seconds, log scale)",
        xticks=qubits,
    )
    axis.grid(True, which="both", linestyle=":", alpha=0.5)
    axis.legend()
    figure.tight_layout()
    path = output / "runtime_comparison.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path, timings


def plot_measurement_limitation(
    values: np.ndarray, output_dir: Path | str | None = None, shots: int = 1024
) -> Path:
    """Compare sampled QFT measurement counts with exact probabilities."""
    output = _prepare_output(output_dir)
    signal = np.asarray(values, dtype=complex)
    if signal.ndim != 1 or signal.size == 0 or (signal.size & (signal.size - 1)):
        raise ValueError("values must be a non-empty vector with power-of-two length")
    if shots < 1:
        raise ValueError("shots must be positive")
    norm = np.linalg.norm(signal)
    if np.isclose(norm, 0.0):
        raise ValueError("values must have a non-zero norm")

    normalized = signal / norm
    state = Statevector(normalized).evolve(qft(int(np.log2(signal.size))))
    counts = state.sample_counts(shots=shots)
    probabilities = np.abs(state.data) ** 2
    bins = np.arange(signal.size)
    measured = np.array(
        [
            sum(count for bitstring, count in counts.items() if int(bitstring, 2) == index)
            for index in bins
        ]
    )

    figure, axis = plt.subplots(figsize=(9, 4.8))
    width = 0.38
    axis.bar(bins - width / 2, probabilities, width=width, label="Exact |amplitude|²")
    axis.bar(
        bins + width / 2,
        measured / shots,
        width=width,
        label=f"Measurement frequency ({shots} shots)",
    )
    axis.set(
        title="QFT Measurement: Probabilities, Not Complex Amplitudes",
        xlabel="Measured frequency bin",
        ylabel="Probability",
        xticks=bins,
    )
    axis.legend()
    axis.grid(axis="y", linestyle=":", alpha=0.5)
    figure.tight_layout()
    path = output / "measurement_probabilities.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path
