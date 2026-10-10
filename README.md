# Black-Scholes Hamiltonian simulation

This project implements Section III of *Efficient Hamiltonian Simulation for Solving Option Price Dynamics* for a European put option.

The implementation in [`black_scholes_quantum.py`](black_scholes_quantum.py) follows the paper’s pipeline:

1. Encode the duplicated payoff on a periodic log-price grid, Eqs. (14)-(20).
2. Transform to momentum space with `qml.QFT`.
3. Apply the bounded even polynomial through `qml.qsvt` with PennyLane’s `qml.FABLE` block encoding.
4. Apply the commuting drift phase and inverse QFT.
5. Measure and post-select the FABLE/QSVT encoding ancillas.
6. Recover the discrete price from the selected physical half-grid probability and payoff normalization, Eq. (25).

The NumPy path is only a classical verification reference. `pennylane_qnode()` constructs and runs the actual quantum circuit; it does not compute or load the final evolved state. The payoff itself is the algorithm’s input state and is prepared with PennyLane’s `qml.StatePrep`, which PennyLane can decompose into elementary gates when the backend does not support native state preparation.

## Run

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Example:

```python
from black_scholes_quantum import BlackScholesConfig, BlackScholesSimulator

sim = BlackScholesSimulator(BlackScholesConfig(n_qubits=8))
result = sim.run(time=1.0, spot=50.0)
print(result.price, result.success_probability)

qnode = sim.pennylane_qnode(shots=None)  # exact probabilities on lightning.qubit
probabilities = qnode()
quantum_result = sim.price_from_pennylane_probabilities(probabilities, time=1.0)
print(quantum_result.price)
```

The existing notebook now includes the PennyLane QFT/QSVT circuit entry point. `sim.run()` remains available only for classical cross-checks against shot-based quantum results.

The physical stock-price grid is the first half of the duplicated periodic grid. The reported price uses the joint probability of the single matching point in that half; it does not add the reflected copy. On hardware, that point probability is too small for reliable direct sampling at modest shot counts, so use amplitude estimation for a useful finite-shot price estimate.
