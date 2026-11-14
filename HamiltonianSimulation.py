import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Efficient Hamiltonian Simulation
    """)
    return


@app.cell
def _():
    import warnings
    warnings.filterwarnings("ignore")

    import pennylane as qp
    from jax import numpy as jnp
    import jax
    jax.config.update("jax_platform_name", "cpu")
    print(jax.default_backend())

    # ---- all library imports live here; add new imports in this cell only ----
    import numpy as np
    import polars as pl
    import altair as alt
    from collections import Counter
    from pennylane.drawer import tape_text
    import pennylane.estimator as qre
    from black_scholes_quantum import BlackScholesConfig, BlackScholesSimulator
    import matplotlib.pyplot as plt


    return (
        BlackScholesConfig,
        BlackScholesSimulator,
        Counter,
        alt,
        jax,
        jnp,
        np,
        pl,
        qp,
        qre,
        tape_text,
    )


@app.cell
def _():
    # Parameters
    s_max = 150     # Maximum value of stock price
    K = 50          # Strike price of the option contract
    r = 0.04        # Risk-free interest rate
    T = 1           # Maturity of the option contract
    sigma = 0.2     # Volatility of the stock price process
    return K, T, r, s_max


@app.cell
def _(jnp, s_max):
    # The number of qubits
    qubits_num = 5

    N_x = 2 ** qubits_num  # Number of grid points in the x-direction

    x_max = 2 * jnp.log(s_max)
    delta_x = 2 * x_max / (N_x - 1)
    x = - 2 * jnp.log(s_max) + delta_x * jnp.arange(N_x)
    sv = jnp.exp(x + x_max / 2)
    e_x = jnp.exp(x)
    return N_x, delta_x, e_x, qubits_num, sv, x, x_max


@app.cell
def _(alt, e_x, mo, np, pl, sv, x):
    df = pl.DataFrame({
        "x": np.asarray(x),
        "sv": np.asarray(sv),
        "e_x": np.asarray(e_x),
    })
    df_long = df.unpivot(index=["x"], on=["sv", "e_x"],
                        variable_name="variable", value_name="value")

    _chart = (
        alt.Chart({"values": df_long.to_dicts()})
        .mark_line()
        .encode(
            x=alt.X("x:Q", title="x"),
            y=alt.Y("value:Q", title="value", scale=alt.Scale(type="log")),
            color=alt.Color("variable:N", title="variable"),
            tooltip=["x:Q", "value:Q", "variable:N"],
        )
        .properties(width="container", height=280, title="Stock grid transforms")
        .interactive()
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(K, N_x, jnp, x, x_max):
    N_max = int(jnp.ceil((jnp.log(K) / (2*x_max) + 1/4) * (N_x - 1)))
    print(N_max, x.shape)
    print(jnp.argmin(jnp.abs(x)))
    return (N_max,)


@app.cell
def _(K, N_max, N_x, delta_x, jnp, s_max, sv, x):
    # Final payoff as the initial condition
    #V_p = jnp.maximum(K - jnp.exp(x), 0.0)
    payoff = jnp.maximum(K - jnp.exp(jnp.log(1/s_max) + delta_x * jnp.arange(N_x/2)), 0.0)

    # The coefficient of the initial state
    coeff = jnp.zeros_like(x)
    coeff = coeff.at[:N_x//2].set(payoff)
    coeff = coeff.at[N_x//2:].set(payoff[::-1])

    # Normalize the coefficient
    coeff = coeff / jnp.linalg.norm(coeff)
    coeff = coeff.at[coeff < 1.0e-6].set(1.0e-6)
    print(jnp.dot(coeff, coeff))
    print(jnp.dot(coeff[:N_x//2], coeff[:N_x//2]))
    print("half normal:", jnp.linalg.norm(coeff[:N_x//2]) * jnp.sqrt(2),  "\n"
            "entry normal:", jnp.linalg.norm(coeff))
    print(sv[N_max])
    return coeff, payoff


@app.cell
def _(N_x, alt, mo, np, payoff, pl, sv):
    df_coeff = pl.DataFrame({
        "x": np.asarray(sv[:N_x//2]),
        "coeff": np.asarray(payoff[:N_x//2]),
    })

    _chart = (
        alt.Chart({"values": df_coeff.to_dicts()})
        .mark_line()
        .encode(
            x=alt.X("x:Q", title="Stock price"),
            y=alt.Y("coeff:Q", title="Payoff amplitude"),
            tooltip=["x:Q", "coeff:Q"],
        )
        .properties(width="container", height=280, title="Initial state coefficients")
        .interactive()
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(N_x, coeff, jnp, qp, qubits_num):
    # Symmetric state loading via QROMStatePreparation: encode only the normalized
    # first half of the palindromic coeff with QROM, then mirror it with qubit-0
    # H + CNOTs. This halves the data QROM must encode vs the full 512-vector.
    _phi = coeff[:N_x // 2] / jnp.linalg.norm(coeff[:N_x // 2])
    wires = qp.registers({'sym_wire': 1, 'state_wires': qubits_num - 1, 'prec_wires': 13, 'work_wires': 1})
    # Configure the wire layout with qp.registers:
    #   sym_wire    : 1 wire,  symmetrization (mirror) qubit
    #   state_wires : qubits_num-1 wires, load the half-state via QROM
    #   prec_wires  : 13 wires, QROM precision register
    #   work_wires  : 1 wire,  QROM work register
    dev = qp.device('lightning.qubit', wires=wires['sym_wire'] + wires['state_wires'] + wires['prec_wires'] + wires['work_wires'])
    shots = 10000

    @qp.simplify
    @qp.qnode(dev)
    def circuit():
        qp.QROMStatePreparation(_phi, wires['state_wires'], wires['prec_wires'], wires['work_wires'])
        qp.Hadamard(wires=wires['sym_wire'])
        for j in wires['state_wires']:
            qp.CNOT(wires=[wires['sym_wire'][0], j])
        return qp.probs(wires['sym_wire'] + wires['state_wires'])
    #@qp.decompose(gate_set={"RX", "RY", "RZ", "CZ", "CNOT"}, max_expansion=2)
    prob = circuit()
    prob.shape  # symmetry qubit  # mirrored half -> full palindromic state
    return circuit, prob


@app.cell
def _(alt, coeff, mo, np, pl, prob, x):
    # state and df_coeff show in a figure
    df_figure = pl.DataFrame({
        "x": np.asarray(x),
        "coeff": np.asarray(coeff)**2,  # target probability
        "prob": np.asarray(prob),
    })
    df_figure_long = df_figure.unpivot(index=["x"], on=["coeff", "prob"],
                                     variable_name="source", value_name="probability")

    _chart = (
        alt.Chart({"values": df_figure_long.to_dicts()})
        .mark_line()
        .encode(
            x=alt.X("x:Q", title="x"),
            y=alt.Y("probability:Q", title="Probability"),
            color=alt.Color("source:N", title="Source"),
            tooltip=["x:Q", "probability:Q", "source:N"],
        )
        .properties(
            width="container",
            height=300,
            title="QROM-prepared probability vs target probability",
        )
        .interactive()
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(circuit, qp):
    print(qp.draw(circuit)())
    return


@app.cell
def _(circuit, qp):
    print(qp.specs(circuit)())
    return


@app.cell
def _(circuit, qre):
    highlvl_gateset = {
        "RX","RY","RZ",
        "Toffoli",
        "X","Y","Z",
        "Adjoint(S)","Adjoint(T)",
        "Hadamard","S","CNOT","T",
    }

    lowlvl_gateset = {"Hadamard", "S", "CNOT", "T"}

    high_level_es = qre.estimate(circuit, gate_set=highlvl_gateset)()
    low_level_es = qre.estimate(circuit, gate_set=lowlvl_gateset)()

    print(f"High-level gateset: \n{high_level_es}")
    print(f"Low-level gateset: \n{low_level_es}")
    return


@app.cell
def _(Counter, coeff, qp, qubits_num, tape_text):
    # Decompose StatePrep into elementary gates (official path: StatePrep -> Mottonen -> RY/CNOT)
    with qp.tape.QuantumTape() as _tape:
        qp.StatePrep(coeff, wires=range(qubits_num))
    _expanded = _tape.expand(depth=10)
    ops = _expanded.operations
    print(f'Total gates: {len(ops)}')
    print('Gate types:', dict(Counter((op.name for op in ops))))
    print()
    print(tape_text(_expanded, wire_order=range(qubits_num)))
    return


@app.cell
def _(coeff, qp, qubits_num):
    # Resource estimation of the state preparation (decomposed: StatePrep -> Mottonen -> RY/CNOT)
    with qp.tape.QuantumTape() as _tape:
        qp.StatePrep(coeff, wires=range(qubits_num))
    _expanded = _tape.expand(depth=10)
    res = _expanded.specs['resources']
    n_1q = res.gate_sizes.get(1, 0)
    n_2q = res.gate_sizes.get(2, 0)
    print('=' * 48)
    print('State Preparation Resource Estimation')
    print('=' * 48)
    print(f'Qubits              : {qubits_num}')
    print(f'Total gates         : {res.num_gates}')
    print(f'  1-qubit gates     : {n_1q}  (RY)')
    print(f'  2-qubit gates     : {n_2q}  (CNOT)')
    print(f'Circuit depth       : {res.depth}')
    print(f'Gate breakdown      : {dict(res.gate_types)}')
    print('=' * 48)
    return


@app.cell
def _(N_x, coeff, jnp, qp, qubits_num):
    # Exploit palindromic symmetry: coeff[i] == coeff[N_x-1-i].
    # Prepare the first half on (qubits_num-1) qubits, then symmetrize with 1 H + (qubits_num-1) CNOTs.
    # This roughly halves the CNOTs vs full 10-qubit Mottonen StatePrep.
    _phi = coeff[:N_x // 2]
    _phi = _phi / jnp.linalg.norm(_phi)  # normalized half-state
    sym_dev = qp.device('default.qubit', wires=qubits_num)
      # 9-wire device -> state() is 2^9, matching coeff
    @qp.qnode(sym_dev)
    def sym_state_prep():
        qp.StatePrep(_phi, wires=list(range(1, qubits_num)))
        qp.Hadamard(wires=0)  # 9-qubit Mottonen
        for j in range(1, qubits_num):  # symmetry qubit
            qp.CNOT(wires=[0, j])
        return qp.state()  # controlled bit-flip -> mirror amplitudes
    sym_state = sym_state_prep()
    print('symmetric prep matches coeff:', bool(jnp.allclose(sym_state, coeff, atol=1e-05)))
    with qp.tape.QuantumTape() as t_orig:
        qp.StatePrep(coeff, wires=range(qubits_num))
    with qp.tape.QuantumTape() as t_sym:
    # Resource comparison: full Mottonen vs symmetric construction
        qp.StatePrep(_phi, wires=list(range(1, qubits_num)))
        qp.Hadamard(wires=0)
        for j in range(1, qubits_num):
            qp.CNOT(wires=[0, j])
    ro, so = (t_orig.expand(depth=10), t_sym.expand(depth=10))
    print(f'Original : {ro.specs['resources'].num_gates} gates, depth {ro.specs['resources'].depth}, {dict(ro.specs['resources'].gate_types)}')
    print(f'Symmetric: {so.specs['resources'].num_gates} gates, depth {so.specs['resources'].depth}, {dict(so.specs['resources'].gate_types)}')
    return


@app.cell
def _(N_x, coeff, np, qp, qubits_num):
    # --- Piecewise-constant (zero-order polynomial) state loading ---
    # coeff is a smooth payoff. Approximate it by P = 2^k constant pieces, each
    # represented by its L2-optimal segment mean. Load the P weights with Mottonen
    # on the top k qubits; the lower (qubits_num-k) qubits are uniform within a
    # piece -> just Hadamards. Cost O(P) vs O(2^N) for full Mottonen.
    coeff_np = np.array(coeff)

    def pw_resources_and_fid(P):
        k = int(np.log2(_P))
        size = N_x // _P
        c_raw = np.array([coeff_np[p * size:(p + 1) * size].mean() for p in range(_P)])  # = ||piecewise-approx||^2 (L2 projection)
        fid = float(size * np.sum(c_raw ** 2))
        c = c_raw / np.linalg.norm(c_raw)
        with qp.tape.QuantumTape() as t:
            qp.StatePrep(c, wires=list(range(k)))
            for w in range(k, qubits_num):
                qp.Hadamard(wires=w)
        r = t.expand(depth=10).specs['resources']
        return (fid, r.num_gates, r.gate_types.get('CNOT', 0), r.depth)
    print(f'{'P':>5} {'fidelity':>10} {'gates':>7} {'CNOT':>6} {'depth':>7}')
    print('-' * 45)
    for _P in [4, 8, 16]:
        f, g, c, d = pw_resources_and_fid(_P)
        print(f'{_P:>5} {f:>10.5f} {g:>7} {c:>6} {d:>7}')
    # full Mottonen reference on qubits_num wires
    with qp.tape.QuantumTape() as t_full:
        qp.StatePrep(coeff_np, wires=range(qubits_num))
    r_full = t_full.expand(depth=10).specs['resources']
    c_full = r_full.gate_types.get('CNOT', 0)
    print(f'{'full':>5} {1.0:>10.5f} {r_full.num_gates:>7} {c_full:>6} {r_full.depth:>7}   <- full Mottonen')
    return (coeff_np,)


@app.cell
def _(N_x, coeff, coeff_np, jnp, np, qp, qubits_num):
    # Chosen configuration: P=16 -> piecewise-constant approximation state load
    # Each element of piece p gets amplitude piece_mean[p]/sqrt(size); this is exactly
    # the tensor product (StatePrep on top k qubits) x (Hadamards on lower qubits).
    # P=16 (k=4, size=2) gives ~99.6% fidelity with only 14 CNOTs; P needs to be
    # >= 16 to keep the end-to-end QFT/QSVT error budget acceptable.
    _P = 16
    k = int(np.log2(_P))
    size = N_x // _P
    piece_mean = np.array([coeff_np[p * size:(p + 1) * size].mean() for p in range(_P)])
    piece_mean = piece_mean / np.linalg.norm(piece_mean)
    pw_dev = qp.device('default.qubit', wires=qubits_num)

    @qp.qnode(pw_dev)
    def pw_state_prep():
        qp.StatePrep(piece_mean, wires=list(range(k)))  # top k MSB qubits: piece weights
        for w in range(k, qubits_num):
            qp.Hadamard(wires=w)  # lower qubits: uniform within piece
        return qp.state()
    pw_state = pw_state_prep()
    fid = float(jnp.abs(jnp.sum(jnp.conj(pw_state) * coeff)) ** 2)
    print(f'P={_P}: overlap with full coeff = {fid:.5f}')  # overlap with true coeff
    pw_amp = np.repeat(np.asarray(piece_mean) / np.sqrt(size), size)
    # reconstitute the full approximated amplitude vector (for plotting/verification)
    print('||reconstituted approx||^2 =', float(np.sum(pw_amp ** 2)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hamiltonian
    """)
    return


@app.cell
def _(N_x, delta_x, jnp):
    from jax.scipy.linalg import expm
    # Option parameters
    rate = 0.04
    sigma_1 = 0.2
    maturity = 1.0
    eigenvalues = jnp.sin(2 * jnp.pi * jnp.arange(N_x) / N_x) / delta_x
    eigen_diag = jnp.diag(eigenvalues)
    Hamiltonian = 0.5 * sigma_1 ** 2 * eigen_diag ** 2 + rate * jnp.eye(N_x)
    exp_Hamiltonian = expm(-Hamiltonian * maturity)
    print(exp_Hamiltonian)
    Hamiltonian.shape
    return Hamiltonian, exp_Hamiltonian, maturity, sigma_1


@app.cell
def _(Hamiltonian, alt, maturity, mo, np, pl):
    # polynomial fitting for QSVT of exp(-maturity*Hamiltonian)
    # BlockEncode normalizes the matrix, so the poly sees singular values in (-1,1).
    # Fit the EVEN target exp(-maturity*m*|x|) on the normalized domain, x in [-1,1].
    m = float(np.max(np.abs(np.diag(Hamiltonian))))  # spectral-norm scale BlockEncode applies
    xs = np.linspace(-1.0, 1.0, 1001)
    cheby_coeff = np.polynomial.chebyshev.chebfit(xs, np.exp(-maturity * m * np.abs(xs)), 40)
    cheby_coeff[1::2] = 0.0  # QSVT requires even parity
    #poly_coeff = np.polynomial.chebyshev.cheb2poly(cheby_coeff)
    poly_coeff = np.polynomial.polynomial.polyfit(xs, np.exp(-maturity * m * np.abs(xs)), 60)
    poly_coeff[1::2] = 0.0  # QSVT requires even parity

    # coefficient values vs term index (Chebyshev vs power basis)
    coef_df = pl.DataFrame({
        "kind": ["cheby"] * cheby_coeff.size + ["poly"] * poly_coeff.size,
        "index": list(range(cheby_coeff.size)) + list(range(poly_coeff.size)),
        "value": np.concatenate([cheby_coeff, poly_coeff]),
    })
    _chart = (
        alt.Chart({"values": coef_df.to_dicts()})
        .mark_point(size=35)
        .encode(
            x=alt.X("index:Q", title="Term index"),
            y=alt.Y("value:Q", title="Coefficient"),
            color=alt.Color("kind:N", title="Basis"),
            tooltip=["kind:N", "index:Q", "value:Q"],
        )
        .properties(width="container", height=300, title="QSVT polynomial coefficients")
        .interactive()
    )
    mo.ui.altair_chart(_chart)
    return m, poly_coeff


@app.cell
def _(
    Hamiltonian,
    N_x,
    alt,
    exp_Hamiltonian,
    m,
    mo,
    np,
    pl,
    poly_coeff,
    qp,
    qubits_num,
):
    # angle of polynomial
    angle_pl = qp.poly_to_angles(poly_coeff, "QSVT")
    print(angle_pl)
    qsvt_H = qp.matrix(qp.qsvt(Hamiltonian/m, poly_coeff, encoding_wires=range(2*qubits_num+1),
                        block_encoding="embedding"))
    print(qsvt_H.real[:N_x, :N_x])
    # diagonal SSR as before (normalized by max diag, squared, NOT divided by N)
    diag_ssr = np.linalg.norm(abs(exp_Hamiltonian.diagonal()-qsvt_H.real[:N_x, :N_x].diagonal())
                             / exp_Hamiltonian.diagonal().max())**2
    # relative Frobenius error of the full 32x32 block (scale- and size-invariant)
    fro_err = np.linalg.norm(exp_Hamiltonian[:N_x,:N_x] - qsvt_H.real[:N_x,:N_x]) \
            / np.linalg.norm(exp_Hamiltonian[:N_x,:N_x])
    print("diag SSR (max-diag normalized):", diag_ssr)
    print("relative Frobenius error:  ", fro_err)

    diag_df = pl.DataFrame({
        "index": list(range(N_x)) * 2,
        "source": ["exp"] * N_x + ["qsvt"] * N_x,
        "value": np.concatenate([
            np.asarray(exp_Hamiltonian.diagonal()),
            np.asarray(qsvt_H.real[:N_x, :N_x].diagonal()),
        ]),
    })
    _chart = (
        alt.Chart({"values": diag_df.to_dicts()})
        .mark_line(point=True)
        .encode(
            x=alt.X("index:Q", title="Diagonal index"),
            y=alt.Y("value:Q", title="Diagonal value", scale=alt.Scale(domain=[0.91, 0.97])), 
            color=alt.Color("source:N", title="Source"),
            tooltip=["index:Q", "value:Q", "source:N"],
        )
        .properties(width="container", height=300, title="Exact vs QSVT diagonal")
        .interactive()
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _():
    #print(jnp.equal(coeff **2, prob))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Paper-aligned Hamiltonian simulation
    """)
    return


@app.cell
def _(jax, jnp):
    def black_scholes_put_price(stock_price, strike, rate, volatility, maturity):
        """Return the Black-Scholes European put price for scalar or array spot values."""
        stock_price = jnp.asarray(stock_price)
        if maturity == 0:
            return jnp.maximum(strike - stock_price, 0.0)
        d1 = (
            jnp.log(stock_price / strike)
            + (rate + 0.5 * volatility**2) * maturity
        ) / (volatility * jnp.sqrt(maturity))
        d2 = d1 - volatility * jnp.sqrt(maturity)
        return (
            strike * jnp.exp(-rate * maturity) * jax.scipy.stats.norm.cdf(-d2)
            - stock_price * jax.scipy.stats.norm.cdf(-d1)
        )


    return (black_scholes_put_price,)


@app.cell
def _(BlackScholesConfig, BlackScholesSimulator, K, T, r, s_max, sigma_1):
    _config = BlackScholesConfig(n_qubits=6, s_max=s_max, strike=K, rate=r, volatility=sigma_1, maturity=T)
    simulator = BlackScholesSimulator(_config)
    spot = simulator.stock_grid[:simulator.size // 2][0]
    result = simulator.run(time=T, spot=spot)
    print(f'grid points: {simulator.size}')
    print(f'ancilla |0> success probability: {result.success_probability:.6f}')
    print(f'discrete put price near S={spot}: {result.price:.6f}')
    return simulator, spot


@app.cell
def _(K, T, black_scholes_put_price, r, sigma_1, spot):
    # Black-Scholes analytic put price at the selected spot.
    put_price = black_scholes_put_price(spot, K, r, sigma_1, T)
    print(f"Black-Scholes put price near S={spot}: {put_price:.6f}")
    return


@app.cell
def _(T, qp, simulator, spot):
    # Positive-momentum convention: payoff → inverse QFT → H-control / QSVT-vs-conjugate / H-control → drift → QFT; post-select final control measurement 0 and FABLE ancillas 0 to select (U + U†)/2.
    qnode = simulator.pennylane_qnode(time=T, shots=None, qsvt_degree=50)
    probabilities = qnode()
    quantum_result = simulator.price_from_pennylane_probabilities(probabilities, spot=spot, time=T)
    print(f"QSVT real-part/FABLE post-selection probability: {quantum_result.success_probability:.6f}")
    print(f"physical grid uses indices 0..{simulator.size // 2 - 1} of {simulator.size} total points")
    print(f"price from physical half-grid point at S={spot}: {quantum_result.price:.6f}")
    print(qp.draw(qnode)())
    return probabilities, qnode


@app.cell
def _(qnode, qre):
    print(qre.estimate(qnode)())
    return


@app.cell
def _():
    #print(qp.draw(qp.decompose(qnode, max_expansion=10))())
    return


@app.cell
def _(T, alt, black_scholes_put_price, mo, np, pl, probabilities, simulator):
    # Compare the final put-value function on the simulator's physical half-grid.
    # Reuse the statevector probabilities from the QSVT QNode above (shots=None).
    _physical_size = simulator.size // 2
    _stock_values = simulator.stock_grid[:_physical_size]
    _config = simulator.config
    _time_value = T
    vol = _config.volatility
    # Black–Scholes analytic value, evaluated at exactly the same stock points.
    rate_value = _config.rate
    strike_value = _config.strike
    analytic_values = np.asarray(
        black_scholes_put_price(
            _stock_values, strike_value, rate_value, vol, _time_value
        )
    )
    discrete_state = simulator.selected_state(_time_value)
    _discrete_values = np.sqrt(simulator.normalization) * np.abs(discrete_state[:_physical_size])
    qsvt_joint_probabilities = np.asarray(probabilities, dtype=float)[:simulator.size]
    qsvt_values = np.sqrt(simulator.normalization * np.maximum(qsvt_joint_probabilities[:_physical_size], 0.0))
    value_df = pl.DataFrame({'Stock price': np.tile(_stock_values, 3), 'Option value': np.concatenate([analytic_values, _discrete_values, qsvt_values]), 'Method': ['Black–Scholes analytic'] * _physical_size + ['Discrete numerical'] * _physical_size + ['QSVT circuit'] * _physical_size})
    # Exact discrete simulator reference and QSVT circuit result. The price at each
    # grid point is sqrt(normalization × joint post-selection probability).
    _chart = (
        alt.Chart({"values": value_df.to_dicts()})
        .mark_line(point=True)
        .encode(
            x=alt.X("Stock price:Q", title="Stock price"),
            y=alt.Y("Option value:Q", title="Put value"),
            color=alt.Color("Method:N", title="Method"),
            tooltip=["Stock price:Q", "Option value:Q", "Method:N"],
        )
        .properties(
            width="container", height=360,
            title="European put value function at maturity",
        )
        .interactive()
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(
    BlackScholesConfig,
    BlackScholesSimulator,
    K,
    T,
    alt,
    black_scholes_put_price,
    mo,
    np,
    pl,
    r,
    s_max,
    sigma_1,
):
    qsvt_degree = 40
    _time_value = T
    curve_rows = []
    _reference_simulator = BlackScholesSimulator(BlackScholesConfig(n_qubits=8, s_max=s_max, strike=K, rate=r, volatility=sigma_1, maturity=T))
    _reference_stock = np.linspace(_reference_simulator.stock_grid[0], _reference_simulator.stock_grid[_reference_simulator.size // 2 - 1], 600)
    _reference_value = np.asarray(
        black_scholes_put_price(
            _reference_stock, K, r, sigma_1, _time_value
        ),
        dtype=np.float64,
    )
    curve_rows.append(pl.DataFrame({'Stock price': _reference_stock, 'Put value': _reference_value, 'Curve': ['Black–Scholes analytic'] * len(_reference_stock)}))
    for _data_qubits in range(4, 8):
        bit_simulator = BlackScholesSimulator(BlackScholesConfig(n_qubits=_data_qubits, s_max=s_max, strike=K, rate=r, volatility=sigma_1, maturity=T))
        bit_probabilities = np.asarray(bit_simulator.pennylane_qnode(time=_time_value, shots=None, qsvt_degree=qsvt_degree)(), dtype=float)
        _physical_size = bit_simulator.size // 2
        bit_stock = bit_simulator.stock_grid[:_physical_size]
        bit_joint_probabilities = bit_probabilities[:bit_simulator.size]
        bit_values = np.sqrt(bit_simulator.normalization * np.maximum(bit_joint_probabilities[:_physical_size], 0.0))
        curve_rows.append(pl.DataFrame({'Stock price': bit_stock, 'Put value': bit_values, 'Curve': [f'QSVT · {_data_qubits} qubits'] * _physical_size}))
    bit_curve_df = pl.concat(curve_rows)
    _chart = (
        alt.Chart({"values": bit_curve_df.to_dicts()})
        .mark_line()
        .encode(
            x=alt.X("Stock price:Q", title="Stock price"),
            y=alt.Y("Put value:Q", title="Put value"),
            color=alt.Color("Curve:N", title="Curve"),
            detail="Curve:N",
            tooltip=["Stock price:Q", "Put value:Q", "Curve:N"],
        )
        .properties(
            width="container", height=360,
            title=f"QSVT option-value curves by data-qubit count (degree {qsvt_degree})",
        )
        .interactive()
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(
    BlackScholesConfig,
    BlackScholesSimulator,
    K,
    T,
    alt,
    black_scholes_put_price,
    mo,
    np,
    pl,
    r,
    s_max,
    sigma_1,
):
    discrete_curve_rows = []
    _time_value = T
    _reference_simulator = BlackScholesSimulator(BlackScholesConfig(n_qubits=10, s_max=s_max, strike=K, rate=r, volatility=sigma_1, maturity=T))
    _reference_stock = np.linspace(_reference_simulator.stock_grid[0], _reference_simulator.stock_grid[_reference_simulator.size // 2 - 1], 800)
    _reference_value = np.asarray(
        black_scholes_put_price(
            _reference_stock, K, r, sigma_1, _time_value
        ),
        dtype=np.float64,
    )
    discrete_curve_rows.append(pl.DataFrame({'Stock price': _reference_stock, 'Put value': _reference_value, 'Curve': ['Black–Scholes analytic'] * len(_reference_stock)}))
    for _data_qubits in range(4, 11):
        grid_simulator = BlackScholesSimulator(BlackScholesConfig(n_qubits=_data_qubits, s_max=s_max, strike=K, rate=r, volatility=sigma_1, maturity=T))
        _physical_size = grid_simulator.size // 2
        _stock_values = grid_simulator.stock_grid[:_physical_size]
        evolved_state = grid_simulator.selected_state(_time_value)
        _discrete_values = np.sqrt(grid_simulator.normalization) * np.abs(evolved_state[:_physical_size])
        discrete_curve_rows.append(pl.DataFrame({'Stock price': _stock_values, 'Put value': _discrete_values, 'Curve': [f'Discrete · {_data_qubits} qubits'] * _physical_size}))
    discrete_curve_df = pl.concat(discrete_curve_rows)
    _chart = (
        alt.Chart({"values": discrete_curve_df.to_dicts()})
        .mark_line()
        .encode(
            x=alt.X("Stock price:Q", title="Stock price"),
            y=alt.Y("Put value:Q", title="Put value"),
            color=alt.Color("Curve:N", title="Curve"),
            detail="Curve:N",
            tooltip=["Stock price:Q", "Put value:Q", "Curve:N"],
        )
        .properties(
            width="container", height=360,
            title="Discrete numerical option-value curves by data-qubit count",
        )
        .interactive()
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
