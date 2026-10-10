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

    def test_pennylane_circuit_keeps_qft_qsvt_drift_inverse_qft_order(self):
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
        self.assertLess(operation_names.index("QFT"), operation_names.index("QSVT"))
        self.assertLess(operation_names.index("QSVT"), operation_names.index("DiagonalQubitUnitary"))
        self.assertLess(operation_names.index("DiagonalQubitUnitary"), operation_names.index("Adjoint(QFT)"))
        probabilities = circuit()
        self.assertEqual(probabilities.shape, (2 ** (2 * small.config.n_qubits + 1),))


if __name__ == "__main__":
    unittest.main()
