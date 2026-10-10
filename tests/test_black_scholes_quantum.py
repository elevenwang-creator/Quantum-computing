import unittest

import numpy as np
import pennylane as qml

from black_scholes_quantum import BlackScholesConfig, BlackScholesSimulator


class BlackScholesSimulatorTests(unittest.TestCase):
    def setUp(self):
        self.config = BlackScholesConfig(
            n_qubits=4, s_max=150.0, strike=50.0, rate=0.04, volatility=0.2, maturity=1.0
        )
        self.simulator = BlackScholesSimulator(self.config)

    def test_grid_and_payoff_follow_paper_equations_15_to_20(self):
        self.assertEqual(self.simulator.grid.size, 2**self.config.n_qubits)
        self.assertAlmostEqual(self.simulator.grid[0], -self.simulator.x_max)
        self.assertAlmostEqual(self.simulator.grid[-1], self.simulator.x_max)
        self.assertTrue(np.allclose(self.simulator.payoff, self.simulator.payoff[::-1]))
        self.assertAlmostEqual(np.linalg.norm(self.simulator.payoff_state), 1.0)

    def test_momentum_eigenvalues_are_the_paper_qft_values(self):
        expected = np.sin(2.0 * np.pi * np.arange(2**self.config.n_qubits) / 2**self.config.n_qubits)
        expected /= self.simulator.dx
        np.testing.assert_allclose(self.simulator.momentum_eigenvalues, expected)

    def test_fourier_helpers_diagonalize_the_paper_momentum_operator(self):
        state = np.array([1.0 + 2.0j, -0.5j, 3.0, 2.0 - 1.0j])
        np.testing.assert_allclose(self.simulator._fourier(state), np.fft.fft(state, norm="ortho"))
        np.testing.assert_allclose(self.simulator._inverse_fourier(state), np.fft.ifft(state, norm="ortho"))

        size = self.simulator.size
        position_momentum = np.zeros((size, size), dtype=complex)
        for j in range(size):
            position_momentum[j, (j + 1) % size] += 1.0
            position_momentum[j, (j - 1) % size] -= 1.0
        position_momentum *= -1j / (2.0 * self.simulator.dx)
        qft = qml.matrix(qml.QFT(wires=range(self.config.n_qubits)))
        momentum = np.diag(self.simulator.momentum_eigenvalues)
        np.testing.assert_allclose(qft @ momentum @ qft.conj().T, position_momentum, atol=1e-10)

    def test_selected_state_uses_hermitian_hamiltonian_phase(self):
        time = 0.35
        p = self.simulator.momentum_eigenvalues
        damping = np.exp(-time * (0.5 * self.config.volatility**2 * p**2 + self.config.rate))
        phase = np.exp(1j * time * (self.config.rate - 0.5 * self.config.volatility**2) * p)
        expected = self.simulator._inverse_fourier(
            damping * phase * self.simulator._fourier(self.simulator.payoff_state)
        )
        np.testing.assert_allclose(self.simulator.selected_state(time), expected, atol=1e-12)

    def test_ancilla_postselection_matches_nonunitary_black_scholes_factor(self):
        time = 0.35
        state = self.simulator.evolve_state(time)
        selected = self.simulator.postselect_ancilla(state)
        expected = self.simulator.selected_state(time)
        np.testing.assert_allclose(selected, expected / np.linalg.norm(expected), atol=1e-10)

    def test_classical_price_is_consistent_with_quantum_position_distribution(self):
        result = self.simulator.run(time=self.config.maturity)
        self.assertGreater(result.success_probability, 0.0)
        self.assertTrue(np.all(result.position_probabilities >= -1e-12))
        self.assertAlmostEqual(float(result.position_probabilities.sum()), 1.0, places=10)
        self.assertGreaterEqual(result.price, 0.0)

    def test_qsvt_postselection_uses_encoding_ancillas_and_physical_half_index(self):
        probabilities = np.zeros(2 ** (2 * self.simulator.n + 1))
        probabilities[0] = 0.16
        probabilities[2] = 0.09
        probabilities[self.simulator.size + 2] = 0.75

        result = self.simulator.price_from_pennylane_probabilities(
            probabilities, spot=self.simulator.stock_grid[2], time=0.2
        )

        self.assertAlmostEqual(result.success_probability, 0.25)
        self.assertAlmostEqual(result.position_probabilities[2], 0.36)
        self.assertAlmostEqual(
            result.price, np.sqrt(0.09 * self.simulator.normalization)
        )

    def test_pennylane_circuit_uses_momentum_basis_order(self):
        small = BlackScholesSimulator(BlackScholesConfig(n_qubits=2))
        circuit = small.pennylane_qnode(time=0.05, qsvt_degree=4)
        tape = circuit.construct([], {})
        operation_names = [op.name for op in tape.operations]
        self.assertIn("StatePrep", operation_names)
        self.assertIn("QFT", operation_names)
        self.assertIn("Adjoint(QFT)", operation_names)
        self.assertIn("QSVT", operation_names)
        self.assertIn("DiagonalQubitUnitary", operation_names)
        self.assertNotIn("QubitUnitary", operation_names)
        self.assertEqual(len(circuit.device.wires), 2 * small.n + 1)
        self.assertLess(operation_names.index("Adjoint(QFT)"), operation_names.index("QSVT"))
        self.assertLess(operation_names.index("QSVT"), operation_names.index("DiagonalQubitUnitary"))
        self.assertLess(operation_names.index("DiagonalQubitUnitary"), operation_names.index("QFT"))
        probabilities = circuit()
        self.assertEqual(probabilities.shape, (2 ** (2 * small.config.n_qubits + 1),))
        diagonal = next(op for op in tape.operations if op.name == "DiagonalQubitUnitary")
        expected_phase = np.exp(
            1j * 0.05 * (small.config.rate - 0.5 * small.config.volatility**2)
            * small.momentum_eigenvalues
        )
        np.testing.assert_allclose(diagonal.data[0], expected_phase, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
