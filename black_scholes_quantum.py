"""PennyLane implementation of the paper's Section III algorithm.

The NumPy methods are classical verification references only.  The
``pennylane_qnode`` method is the quantum algorithm: payoff state preparation,
QFT, QSVT/FABLE block encoding of the non-unitary Black-Scholes factor,
momentum-space drift phase, inverse QFT, and measurement of the encoding
ancillas.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

_QSVT_POLY_SCALE = 1.0 - 1e-10


@dataclass(frozen=True)
class BlackScholesConfig:
    n_qubits: int = 8
    s_max: float = 150.0
    strike: float = 50.0
    rate: float = 0.04
    volatility: float = 0.2
    maturity: float = 1.0
    spot: Optional[float] = None

    def __post_init__(self) -> None:
        if self.n_qubits < 2:
            raise ValueError("n_qubits must be at least 2")
        if min(self.s_max, self.strike, self.volatility) <= 0:
            raise ValueError("s_max, strike, and volatility must be positive")
        if self.maturity < 0:
            raise ValueError("maturity must be non-negative")


@dataclass
class SimulationResult:
    time: float
    state: np.ndarray
    success_probability: float
    position_probabilities: np.ndarray
    stock_grid: np.ndarray
    price: float
    target_index: int


class BlackScholesSimulator:
    """Hamiltonian simulation for a European put under the paper's framework."""

    def __init__(self, config: BlackScholesConfig):
        self.config = config
        self.n = config.n_qubits
        self.size = 2**self.n
        self.x_max = 2.0 * np.log(config.s_max)
        self.dx = 2.0 * self.x_max / (self.size - 1)
        self.grid = -self.x_max + self.dx * np.arange(self.size)
        self.stock_grid = np.exp(self.grid + self.x_max / 2)
        self.momentum_eigenvalues = (
            np.sin(2.0 * np.pi * np.arange(self.size) / self.size) / self.dx
        )
        self.n_max = int(np.ceil((np.log(config.strike) / (2.0 * self.x_max) + 0.25) * (self.size - 1)))
        self.payoff = self._duplicated_payoff()
        self.normalization = float(np.dot(self.payoff, self.payoff))
        self.payoff_state = self.payoff / np.sqrt(self.normalization)

    def _duplicated_payoff(self) -> np.ndarray:
        """Eq. (14) and Eqs. (17)-(20), with periodic reflection."""
        half_grid = -self.x_max / 2.0 + self.dx * np.arange(self.size // 2)
        half_payoff = np.maximum(self.config.strike - np.exp(half_grid), 0.0)
        result = np.zeros(self.size)
        count = min(max(self.n_max + 1, 1), self.size // 2)
        result[:count] = half_payoff[:count]
        result[-count:] = half_payoff[:count][::-1]
        return result

    def _physical_spot_index(self, spot: Optional[float]) -> int:
        """Map a stock price to the non-duplicated half of the log grid."""
        target_spot = spot or self.config.spot or self.config.strike
        physical_grid = self.stock_grid[: self.size // 2]
        if target_spot < physical_grid[0] or target_spot > physical_grid[-1]:
            raise ValueError(
                f"spot must lie in the physical half-grid [{physical_grid[0]:.6g}, {physical_grid[-1]:.6g}]"
            )
        return int(np.argmin(np.abs(physical_grid - target_spot)))

    def _fourier(self, state: np.ndarray) -> np.ndarray:
        return np.fft.fft(state, norm="ortho")

    def _inverse_fourier(self, state: np.ndarray) -> np.ndarray:
        return np.fft.ifft(state, norm="ortho")

    def nonunitary_state(self, time: float, state: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply O(t)=exp[-t(σ² p²/2+rI)] in the momentum basis."""
        vector = self.payoff_state if state is None else np.asarray(state, dtype=complex)
        momentum_state = self._fourier(vector)
        damping = np.exp(-time * (0.5 * self.config.volatility**2 * self.momentum_eigenvalues**2 + self.config.rate))
        return self._inverse_fourier(damping * momentum_state)

    def selected_state(self, time: float, state: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply the full selected propagator O(t) exp(i t r p)."""
        vector = self.payoff_state if state is None else np.asarray(state, dtype=complex)
        momentum_state = self._fourier(vector)
        p = self.momentum_eigenvalues
        damping = np.exp(-time * (0.5 * self.config.volatility**2 * p**2 + self.config.rate))
        drift_rate = self.config.rate - 0.5 * self.config.volatility**2
        return self._inverse_fourier(damping * np.exp(1j * time * drift_rate * p) * momentum_state)

    def evolve_state(self, time: float, state: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the exact embedded state [|0>O|ψ>, |1>sqrt(I-O²)|ψ>]."""
        vector = self.payoff_state if state is None else np.asarray(state, dtype=complex)
        momentum_state = self._fourier(vector)
        p = self.momentum_eigenvalues
        damping = np.exp(-time * (0.5 * self.config.volatility**2 * p**2 + self.config.rate))
        drift_rate = self.config.rate - 0.5 * self.config.volatility**2
        drift = np.exp(1j * time * drift_rate * p)
        selected = self._inverse_fourier(damping * drift * momentum_state)
        complement = self._inverse_fourier(np.sqrt(np.maximum(1.0 - damping**2, 0.0)) * momentum_state)
        return np.concatenate([selected, complement])

    def postselect_ancilla(self, embedded_state: np.ndarray) -> np.ndarray:
        selected = np.asarray(embedded_state, dtype=complex)[: self.size]
        norm = np.linalg.norm(selected)
        if norm == 0:
            raise ValueError("ancilla |0> post-selection has zero probability")
        return selected / norm

    def run(self, time: Optional[float] = None, spot: Optional[float] = None) -> SimulationResult:
        time = self.config.maturity if time is None else time
        embedded = self.evolve_state(time)
        selected = embedded[: self.size]
        success = float(np.vdot(selected, selected).real)
        conditional = np.abs(selected / np.sqrt(success)) ** 2
        target_index = self._physical_spot_index(spot)
        price = float(np.sqrt(success * conditional[target_index] * self.normalization))
        return SimulationResult(time, embedded, success, conditional, self.stock_grid, price, target_index)

    def _qsvt_polynomial(self, time: float, degree: int = 16) -> np.ndarray:
        """Fit the even polynomial used for the dissipative propagator."""
        if degree % 2:
            raise ValueError("QSVT degree must be even")
        alpha = time * self._fable_scale()
        nodes = np.linspace(-1.0, 1.0, max(129, 4 * (degree + 1)))
        chebyshev = np.polynomial.chebyshev.chebfit(nodes, np.exp(-alpha * np.abs(nodes)), degree)
        chebyshev[1::2] = 0.0
        return np.polynomial.chebyshev.cheb2poly(chebyshev) * _QSVT_POLY_SCALE

    def _fable_scale(self) -> float:
        """Scale the dissipative Hamiltonian to PennyLane QSVT's FABLE bound."""
        p = self.momentum_eigenvalues
        b = 0.5 * self.config.volatility**2 * p**2 + self.config.rate
        return float(1.01 * self.size * np.max(np.abs(b)))

    def _damping_matrix(self) -> np.ndarray:
        p = self.momentum_eigenvalues
        b = 0.5 * self.config.volatility**2 * p**2 + self.config.rate
        return np.diag(b / self._fable_scale())

    def pennylane_qnode(
        self, time: Optional[float] = None, shots: Optional[int] = None, qsvt_degree: int = 16
    ) -> Any:
        """Build the PennyLane inverse-QFT → even-QSVT → drift → QFT circuit."""
        try:
            import pennylane as qml
        except ImportError as exc:
            raise ImportError("Install PennyLane to create the quantum QNode: pip install pennylane") from exc
        time = self.config.maturity if time is None else time
        wires = list(range(2 * self.n + 1))
        qsvt_wires = wires
        # PennyLane's FABLE/QSVT projector selects the first 2**n basis
        # states: leading n+1 encoding wires are zero, trailing n wires hold j.
        data_wires = qsvt_wires[-self.n :]
        dev = qml.device("lightning.qubit", wires=wires)
        damping_matrix = self._damping_matrix()
        polynomial = self._qsvt_polynomial(time, qsvt_degree)
        drift_rate = self.config.rate - 0.5 * self.config.volatility**2
        drift = np.exp(1j * time * drift_rate * self.momentum_eigenvalues)
        with qml.QueuingManager.stop_recording():
            qsvt = qml.qsvt(
                damping_matrix,
                polynomial,
                encoding_wires=qsvt_wires,
                block_encoding="fable",
            )

        @qml.qnode(dev, shots=shots)
        def circuit():
            # Eq. (20): the input payoff is loaded as an n-qubit state.
            qml.StatePrep(self.payoff_state, wires=data_wires)
            if time == 0:
                return qml.probs(wires=wires)
            qml.adjoint(qml.QFT)(wires=data_wires)
            qml.apply(qsvt)
            qml.DiagonalQubitUnitary(drift, wires=data_wires)
            qml.QFT(wires=data_wires)
            return qml.probs(wires=wires)

        return circuit

    def price_from_pennylane_probabilities(
        self,
        probabilities: np.ndarray,
        spot: Optional[float] = None,
        time: Optional[float] = None,
    ) -> SimulationResult:
        """Post-select QSVT/FABLE encoding ancillas and estimate the price.

        ``probabilities`` must be returned by ``pennylane_qnode`` with all
        wires measured. Post-selection sets the QSVT/FABLE encoding ancillas to zero.
        """
        probabilities = np.asarray(probabilities, dtype=float)
        time = self.config.maturity if time is None else time
        expected_size = 2 ** (2 * self.n + 1)
        if probabilities.shape != (expected_size,):
            raise ValueError(f"expected probabilities with shape {(expected_size,)}, got {probabilities.shape}")
        # FABLE's leading n+1 encoding ancillas are zero in the selected block.
        data_probabilities = probabilities[: self.size]
        success = float(data_probabilities.sum())
        if success <= 0:
            raise ValueError("no shots survived QSVT/FABLE ancilla post-selection")
        conditional = data_probabilities / success
        target_index = self._physical_spot_index(spot)
        price = float(np.sqrt(success * conditional[target_index] * self.normalization))
        return SimulationResult(
            time=time,
            state=np.asarray([], dtype=complex),
            success_probability=success,
            position_probabilities=conditional,
            stock_grid=self.stock_grid,
            price=price,
            target_index=target_index,
        )
