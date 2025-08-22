import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, sqrtm
from qiskit import transpile, QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
from qutip import Qobj, mesolve, Options      # core QuTiP classes and solver

USE_TROTTER = 0  # if set to 0, we will use Strang splitting instead
USE_DYNAMIC_DECOUPLE = 0  # only for strang splitting, this is to reduce noise for increased fidelity
USE_MULTIPLE_QUBITS = 1  # for simulating multiple qubits interaction baths


# --- 1. Define the Basic Building Blocks (Pauli Matrices) ---
I = np.array([[1, 0], [0, 1]], dtype=complex)
sigmax = np.array([[0, 1], [1, 0]], dtype=complex)
sigmay = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigmaz = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_plus = np.array([[0, 1], [0, 0]], dtype=complex)  # Physical Lowering Operator (Emission, σ⁻): This action (|1⟩ → |0⟩) is performed by 0.5 * (sigmax + 1j * sigmay)
sigma_minus = np.array([[0, 0], [1, 0]], dtype=complex)  # Physical Raising Operator (Absorption, σ⁺): This action (|0⟩ → |1⟩) is performed by 0.5 * (sigmax - 1j * sigmay)


# --- Section for actual quantum run data ---
# This part simulates getting data from a perfect (noiseless) quantum device
# to compare against our manual simulation.

# We need a function that can create an operator like σx¹σx² that acts on two specific qubits within the larger N-qubit space.
def promote_op_pair(op1_1q, op2_1q, q_idx1, q_idx2, total_qubits):
    """
    Creates an N-qubit operator acting on two specified qubits.
    Example: promote_op_pair(sigmax, sigmax, 0, 1, 4) -> σx¹ ⊗ σx² ⊗ I³ ⊗ I⁴
    """
    op_list = [I] * total_qubits
    op_list[q_idx1] = op1_1q
    op_list[q_idx2] = op2_1q

    full_op = op_list[0]
    for i in range(1, total_qubits):
        full_op = np.kron(full_op, op_list[i])

    return full_op

# Helper function to promote single-qubit operators to the N-qubit space
def promote_op(op_1q, qubit_idx, total_qubits):
    """
    Creates an N-qubit operator from a single-qubit operator.
    Example: promote_op(sigmaz, 0, 4) -> σz¹ ⊗ I² ⊗ I³ ⊗ I⁴
    """
    op_list = [I] * total_qubits
    op_list[qubit_idx] = op_1q

    full_op = op_list[0]
    for i in range(1, total_qubits):
        full_op = np.kron(full_op, op_list[i])

    return full_op


g = 1.0 * 2 * np.pi  # Rabi frequency (2pi so the period is 1)

if not USE_MULTIPLE_QUBITS:
    # --- 2. System Parameters ---
    H = g / 2.0 * sigmax # Hamiltonian

    # Environment interaction rates
    gamma = 0.3  # Relaxation rate
    kappa = 0.2  # Dephasing rate
else:
    # --- 2. System Parameters ---
    N = 4  # Total number of qubits in the chain
    # Initialize the Hamiltonian as a zero matrix of the correct size
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)

    # On-site energies (transition frequencies) for each qubit
    w = [1.0] * N  # For Fig. 2, all frequencies are equal.

    # Interaction strengths (couplings) between adjacent qubits
    # We use a dictionary for clarity: J[(qubit_i, qubit_j)] = strength
    J = {
        (0, 1): 0.45, # Corresponds to J14 in the paper's relabeled Fig. 2 params
        (1, 2): 0.56, # Corresponds to J23
        (2, 3): 0.75  # Corresponds to J34
    }

    # Bath Parameters (Bath A on qubit 0, Bath B on qubit N-1)
    gamma_A = 0.1
    gamma_B = 0.05
    T_A = 0.0
    T_B = 4.0 # Using the temperature from Fig. 2

    # Bose-Einstein distribution for thermal occupations
    # np.finfo(float).eps avoids division by zero if a temperature is 0
    N_th_A = 1 / (np.exp(w[0] / (T_A + np.finfo(float).eps)) - 1)
    N_th_B = 1 / (np.exp(w[N-1] / (T_B + np.finfo(float).eps)) - 1)

# --- 4. Simulation Setup ---
if not USE_MULTIPLE_QUBITS:
    # Initial state: ρ(0) = |1><1|
    #psi0_vec = np.array([[0], [1]], dtype=complex)

    # Initial state: ρ(0) = (|0> + i|1>)/sqrt(2)
    psi0_vec = (1/np.sqrt(2)) * np.array([[1], [1j]], dtype=complex)
    rho0 = psi0_vec @ psi0_vec.conj().T

    # --- COLLAPSE OPERATORS ---
    L1 = np.sqrt(gamma) * sigma_plus      # Relaxation
    L2 = np.sqrt(kappa / 2.0) * sigmaz  # Dephasing
    open_system_ops = [L1, L2]
    closed_system_ops = []
else:
    # Initial state: ground state |00...0>
    psi0_vec = np.zeros(dim, dtype=complex)
    psi0_vec[0] = 1
    rho0 = np.outer(psi0_vec, psi0_vec.conj())

    # Build Hamiltonian from Eq. (5)
    print(f"Building Hamiltonian for {N}-qubit system...")
    # Add on-site energy terms: Σ (ω_k/2) σz_k
    for k in range(N):
        H += (w[k]/2) * promote_op(sigmaz, k, N)

    # Add interaction terms: Σ J_lk (σx_l σx_k + σy_l σy_k)
    # which is equivalent to 2 * J_lk * (σ+_l σ-_k + σ-_l σ+_k)
    for (l, k), J_val in J.items():
        # Directly build the sx_l @ sx_k and sy_l @ sy_k terms
        sx_l_sx_k = promote_op_pair(sigmax, sigmax, l, k, N)
        sy_l_sy_k = promote_op_pair(sigmay, sigmay, l, k, N)
        H += J_val * (sx_l_sx_k + sy_l_sy_k)

    # --- Build Thermal Collapse Operators for the End Qubits ---
    # Bath A (cold) is coupled to qubit 0
    L_A_down = np.sqrt(gamma_A * (N_th_A + 1)) * promote_op(sigma_plus, 0, N)
    L_A_up   = np.sqrt(gamma_A * N_th_A) * promote_op(sigma_minus, 0, N)

    # Bath B (hot) is coupled to qubit N-1
    L_B_down = np.sqrt(gamma_B * (N_th_B + 1)) * promote_op(sigma_plus, N-1, N)
    L_B_up   = np.sqrt(gamma_B * N_th_B) * promote_op(sigma_minus, N-1, N)

    open_system_ops = [L_A_down, L_A_up, L_B_down, L_B_up]
    closed_system_ops = []


if not USE_MULTIPLE_QUBITS:
    n_steps = 200
    step_size = 10.0
else:
    n_steps = 400  # Longer time for N-qubit thermalization
    step_size = 20.0

tlist = np.linspace(0, step_size, n_steps)

# Duration of the gate in the same units
dt = tlist[1] - tlist[0]


# --- build circuit_list ---
circuit_list = []

print("--- Building circuits iteratively to accumulate noise ---")

if not USE_MULTIPLE_QUBITS:
    # --- Single-Qubit Case ---

    # 1. Define the small, constant-sized operation for one time step 'dt'
    U_step = QuantumCircuit(1, name="U_step")
    U_step.rx(g * dt, 0)

    # 2. Start with the initial state at t=0
    current_circ = QuantumCircuit(1)
    current_circ.x(0)  # Prepare |1>
    current_circ.save_density_matrix()
    circuit_list.append(current_circ.copy())

    # 3. Iteratively add one step at a time to build the full evolution
    for i in range(1, n_steps):
        # Add the next small gate to the end of the previous circuit
        current_circ.compose(U_step, inplace=True)

        # Remove the previous 'save' instruction from the middle of the circuit
        if len(current_circ.data) > i + 1:
            current_circ.data.pop(-2)

        # Add a new 'save' instruction at the very end
        current_circ.save_density_matrix()
        circuit_list.append(current_circ.copy())

else:
    # --- N-Qubit Case ---
    print("--- Building N-QUBIT noisy circuits ---")

    # --- a) Build a Custom Quantum Error Channel from our Collapse Operators ---

    # We use a two-step process:
    # 1. Import the necessary classes: QuantumError from aer and Kraus from quantum_info.
    # 2. Build a Kraus channel object from our raw NumPy matrices.
    # 3. Pass the well-defined channel object to the QuantumError constructor.

    if USE_TROTTER:
        from qiskit_aer.noise import QuantumError
        from qiskit.quantum_info import Kraus

        # Create a list of the raw NumPy matrices for the Kraus operators
        dim = 2**N
        L_dag_L_sum = np.zeros((dim, dim), dtype=complex)
        kraus_mats_np = []

        # Create the "jump" operators (M_k for k > 0)
        for L in open_system_ops:
            # The sum of L†L is needed to calculate the no-jump operator
            L_dag_L_sum += L.conj().T @ L
            kraus_mats_np.append(np.sqrt(dt) * L)

        # Create the "no-jump" operator (M_0) by taking the exact matrix square root
        # This ensures the channel is perfectly trace-preserving (CPTP).
        identity_N = np.identity(dim)
        # The argument of sqrtm must be Hermitian, which it is.
        M0_matrix = sqrtm(identity_N - dt * L_dag_L_sum)
        kraus_mats_np.insert(0, M0_matrix)

        # Step 1: Create a QuantumChannel object (Kraus) from the NumPy matrices.
        kraus_channel = Kraus(kraus_mats_np)

        # Step 2: Create the QuantumError object from the channel.
        dissipative_channel = QuantumError(kraus_channel)

        # 1. Define the small, constant-sized operation for one time step 'dt'
        U_step_matrix = expm(-1j * H * dt)
        U_step_gate = Operator(U_step_matrix)
        U_step = QuantumCircuit(N, name="U_step_N")
        U_step.unitary(U_step_gate, range(N), label="unitary_step")

        # 2. Start with the initial state at t=0 (|00...0>)
        current_circ = QuantumCircuit(N)
        current_circ.save_density_matrix()
        circuit_list.append(current_circ.copy())

        # 3. Iteratively add one step at a time
        for i in range(1, n_steps):
            # Add the next small gate to the end of the previous circuit
            current_circ.compose(U_step, inplace=True)

            # Remove the previous 'save' instruction
            if len(current_circ.data) > i: # In this case, it's just i
                current_circ.data.pop(-2)

            # Add a new 'save' instruction at the very end
            current_circ.save_density_matrix()
            circuit_list.append(current_circ.copy())

    else:
        # --- 2. Build the Custom Operations for a Strang Splitting Step ---

        from qiskit.quantum_info import Kraus

        # a) The Unitary part for a HALF step (dt/2)
        U_half_matrix = expm(-1j * H * (dt/2))
        U_half_gate = Operator(U_half_matrix)

        # b) The Dissipative part for a FULL step (dt)
        # We build the Kraus channel that represents D(dt).
        dim = H.shape[0]
        I_N = np.identity(dim)
        L_dag_L_sum = np.zeros((dim, dim), dtype=complex)
        kraus_mats_np = []
        for L in open_system_ops:
            L_dag_L_sum += L.conj().T @ L
            kraus_mats_np.append(np.sqrt(dt) * L) # Jump operators
        M0_matrix = sqrtm(I_N - dt * L_dag_L_sum) # Exact no-jump operator
        kraus_mats_np.insert(0, M0_matrix)
        dissipative_channel = Kraus(kraus_mats_np) # The QuantumChannel for D(dt)

        # c) Assemble the full, accurate U_step circuit
        U_step_strang = QuantumCircuit(N, name="Strang_Step")
        U_step_strang.unitary(U_half_gate, range(N), label="unitary_step")
        U_step_strang.append(dissipative_channel, range(N))
        U_step_strang.unitary(U_half_gate, range(N), label="unitary_step")


        # --- 3. Build Circuits Iteratively ---
        print("--- Building circuits with manual Strang splitting ---")

        current_circ = QuantumCircuit(N)
        current_circ.save_density_matrix()
        circuit_list.append(current_circ.copy())

        # Create a clean 'current_circ' that we will build up without save instructions
        current_circ = QuantumCircuit(N)

        # Iteratively build the full evolution
        for i in range(1, n_steps):
            # Add the next Strang step to our clean evolving circuit
            current_circ.compose(U_step_strang, inplace=True)

            # Create a temporary copy for this specific time step
            temp_circ = current_circ.copy()
            # Add the save instruction ONLY to the temporary copy
            temp_circ.save_density_matrix()
            # Add the finished, saved circuit to our final list
            circuit_list.append(temp_circ)


# --- Visualize a Representative Circuit from the List ---
# We'll draw the circuit after 3 time steps to verify the structure.
# Note: Drawing the final circuit (circuit_list[-1]) is not recommended as it will be too long to read.
if len(circuit_list) > 3:
    circuit_to_draw = circuit_list[3]

    print("\n--- Visualizing the Circuit After 3 Time Steps ---")
    # The .draw('mpl') method returns a matplotlib Figure object.
    # To show it in a script, we must call plt.show().
    # For this to work, you may need to install a package: pip install pylatexenc
    try:
        # Draw the circuit to a matplotlib figure
        circuit_figure = circuit_to_draw.draw('mpl', style='iqx')
        # Add a title to the plot
        circuit_figure.suptitle("Circuit Diagram After 3 Time Steps", fontsize=16)
        # Display the figure in a new window
        plt.show()
    except Exception as e:
        print(f"\nPlotting with 'mpl' failed: {e}. Using text fallback.")
        print(circuit_to_draw.draw('text'))


# --- Analyze Final Circuit Complexity ---
# We will analyze the longest circuit in our list, which corresponds to the final time step.
final_circuit = circuit_list[-1]

# Decompose the circuit to see the fundamental gates if needed
# final_circuit_decomposed = final_circuit.decompose()

print("\n--- Circuit Complexity Analysis (Final Time Step) ---")
print(f"Total Number of Qubits: {final_circuit.num_qubits}")
print(f"Total Gate Count (.size()): {final_circuit.size()}")
print(f"Circuit Depth (.depth()): {final_circuit.depth()}")


print("--- Running Qiskit Aer simulation to get 'hardware' data ---")
from qiskit_aer.noise import NoiseModel, pauli_error, thermal_relaxation_error

# Create an empty noise model
noise_model = NoiseModel(basis_gates=['rx', 'x', 'id']) # Specify the gates our model applies to
basis_gates = noise_model.basis_gates

if not USE_MULTIPLE_QUBITS:
    # Convert gamma/kappa rates to T1/T2 times (standard in Qiskit)
    T1_noise = 1.0 / gamma
    T2_noise_dephasing = 1.0 / kappa # T2 from pure dephasing
    # The total T2 is limited by both T1 and pure dephasing: 1/T2 = 1/(2*T1) + 1/T2_dephasing
    T2_noise = 1.0 / ( (1.0 / (2.0 * T1_noise)) + (1.0 / T2_noise_dephasing) )

    # Add T1 and T2 errors to the RX gate
    # T1 = time for relaxation, T2 = time for dephasing
    errors = thermal_relaxation_error(T1_noise, T2_noise, dt)

    # Add the errors to the noise model, specifying which gates they apply to.
    noise_model.add_all_qubit_quantum_error(errors, ['rx'])
    noise_model.add_all_qubit_quantum_error(errors, ['x'])
else:
    # Apply our custom dissipative channel after every custom 'unitary_step' gate
    noise_model.add_all_qubit_quantum_error(dissipative_channel, ["unitary_step"])
    # noise_model = None

# --- Transpile in the SAME basis that has noise; avoid synthesis that removes rx
backend = AerSimulator(method='density_matrix')

# Transpile all circuits at once for efficiency
transpiled_circs = transpile(circuit_list, backend=backend)#, optimization_level=0)
job = backend.run(transpiled_circs, shots=1, noise_model=noise_model)
result = job.result()

rhos_hw = []
for i in range(len(circuit_list)):
    # The key is "density_matrix" because that's the default label
    # for the save_density_matrix() instruction.
    rho = result.data(i)["density_matrix"]
    rhos_hw.append(rho)

rhos_hw = np.array(rhos_hw)  # shape (n_steps, 2, 2)
print("--- Qiskit simulation complete ---")



# --- 3. Define Dissipator and Initial State ---
def dissipator(rho, collapse_ops):
    """Calculates ONLY the dissipative part of the Lindblad equation."""
    """dissipator is describing non‐unitary “jump” and “no‐jump” processes"""
    dissipator_part = np.zeros_like(rho)
    for L in collapse_ops:
        L_dag = L.conj().T
        L_dag_L = L_dag @ L
        term1 = L @ rho @ L_dag
        anticommutator_term = 0.5 * (L_dag_L @ rho + rho @ L_dag_L)  # to keep total prob = 1
        dissipator_part += (term1 - anticommutator_term)
    return dissipator_part



# --- 5. Run Numerically Stable Simulation ---
def run_stable_simulation(rho_initial, H, tlist, collapse_ops):
    """
    Evolves the density matrix using a stable operator-splitting method.
    The unitary part is evolved exactly, and the dissipative part with Euler.
    """
    dt = tlist[1] - tlist[0]
    rho = rho_initial.copy()

    # This is the rotation operator for a single time-step dt.
    U = expm(-1j * H * dt)
    U_dag = U.conj().T

    # The pi-pulse (X gate) operator
    U_pi = sigmax # In this basis, RX(pi) is just the sigma-X matrix.

    U_half   = expm(-1j * H * (dt/2))
    U_half_dag = U_half.conj().T

    # Unitary for a quarter-step. We evolve for dt/4, flip, evolve for dt/4 again.
    U_quarter = expm(-1j * H * (dt/4))
    U_quarter_dag = U_quarter.conj().T

    population_history = []
    # ### <<< FIX 3: Create a list to store the density matrix at each step ###
    rhos_history = []

    for _ in tlist:
        population_history.append(np.real(rho[1, 1]))
        # ### <<< FIX 3: Append the current density matrix to the history list ###
        # Use .copy() to store the value, not a reference to the changing rho object
        rhos_history.append(rho.copy())

        if USE_TROTTER:
            # This is a 1st-order splitting method (Lie-Trotter splitting)
            # Step A: Evolve the unitary part EXACTLY for one time step.
            rho = U @ rho @ U_dag
            rho += dt * dissipator(rho, collapse_ops)
        else:
            if USE_DYNAMIC_DECOUPLE:
                # --- Dynamical Decoupling Step using Strang Splitting ---
                # --- Hahn Echo Sequence Step ---
                # 1. Evolve everything for the first half-step
                rho = U_half @ rho @ U_half_dag
                rho = rho + (dt/2) * dissipator(rho, collapse_ops) # Half the noise

                # 2. Apply the Pi-Pulse
                rho = U_pi @ rho @ U_pi.conj().T

                # 3. Evolve everything for the second half-step
                rho = U_half @ rho @ U_half_dag
                rho = rho + (dt/2) * dissipator(rho, collapse_ops) # Second half of noise
            else:
                # Strang (2nd-order) symmetric splitting:
                rho = U_half @ rho @ U_half_dag          # half-step Hamiltonian
                rho = rho + dt * dissipator(rho, collapse_ops)  # full dissipator
                rho = U_half @ rho @ U_half_dag          # another half-step Hamiltonian

    # ### <<< FIX 3: Return the history of density matrices ###
    return population_history, np.array(rhos_history)


def run_mesolve_simulation(rho0_np, H_np, tlist, c_ops_np):
    """
    Runs the QuTiP master equation solver for a generic N-qubit system.
    It automatically determines N and sets up e_ops to measure <σz> for each qubit.

    Args:
        rho0_np, H_np: Initial state and Hamiltonian as NumPy arrays.
        tlist: List of times for the evolution.
        c_ops_np: List of collapse operators as NumPy arrays.

    Returns:
        The result object from QuTiP's mesolve.
    """
    # 1. Convert all NumPy inputs to QuTiP's Qobj format
    H_qobj = Qobj(H_np)
    rho0_qobj = Qobj(rho0_np)
    c_ops_qobj = [Qobj(L) for L in c_ops_np]

    # 2. Determine the number of qubits, N, from the Hamiltonian dimension
    dim = H_np.shape[0]
    N = int(np.log2(dim))

    # 3. Dynamically build the list of expectation operators [<σz_0>, <σz_1>, ...]
    print(f"QuTiP mesolve: Setting up e_ops for {N} qubits...")
    e_ops = []

    if N == 1:
        # For a single qubit, we want all three Pauli expectations
        e_ops = [Qobj(sigmax), Qobj(sigmay), Qobj(sigmaz)]
    else:
        for k in range(N):
            # Create the sz operator for the k-th qubit in the N-qubit space
            sz_k_np = promote_op(sigmaz, k, N)
            e_ops.append(Qobj(sz_k_np))

    # 4. Set options to store the full density matrix history
    opts = {"store_states": True}

    # 5. Solve the master equation
    result = mesolve(H_qobj, rho0_qobj, tlist, c_ops_qobj, e_ops=e_ops, options=opts)

    return result


def apply_kraus_map(rho, kraus_ops):
    """Applies a list of Kraus operators to a density matrix."""
    new_rho = np.zeros_like(rho)
    for K in kraus_ops:
        new_rho += K @ rho @ K.conj().T
    return new_rho

def run_kraus_simulation(rho_initial, H, tlist, collapse_ops):
    """
    Evolves the density matrix by directly applying the Kraus map at each step.
    This is the most direct comparison to Qiskit's method.
    """
    dt = tlist[1] - tlist[0]
    rho = np.copy(rho_initial)

    # 1. Build the EXACT Kraus channel for the dissipative part D(dt)
    dim = H.shape[0]
    L_dag_L_sum = np.zeros((dim, dim), dtype=complex)
    kraus_ops_D = []
    for L in collapse_ops:
        L_dag_L_sum += L.conj().T @ L
        kraus_ops_D.append(np.sqrt(dt) * L)
    M0 = sqrtm(np.identity(dim) - dt * L_dag_L_sum)
    kraus_ops_D.insert(0, M0)

    # 2. Pre-calculate the unitary for a full time-step
    U_full = expm(-1j * H * dt)

    # 2. Pre-calculate the unitaries for the Strang splitting
    U_half = expm(-1j * H * (dt/2))

    rhos_history = [rho.copy()]
    for _ in range(len(tlist) - 1):
        if USE_TROTTER:
            # Apply 1st-order Trotter: U(dt) then D(dt)
            rho = U_full @ rho @ U_full.conj().T
            rho = apply_kraus_map(rho, kraus_ops_D)
        else:
            # Apply Strang Splitting using the EXACT Kraus map for the noise
            rho = U_half @ rho @ U_half.conj().T
            rho = apply_kraus_map(rho, kraus_ops_D)
            rho = U_half @ rho @ U_half.conj().T

        rhos_history.append(rho.copy())

    return np.array(rhos_history)


print("Running STABLE simulation for the CLOSED system...")
pop_closed_stable, rhos_closed = run_stable_simulation(rho0, H, tlist, closed_system_ops)

print("Running STABLE simulation for the OPEN system...")
pop_open_stable, rhos_strang = run_stable_simulation(rho0, H, tlist, open_system_ops)

print("Running MESOLVE simulation for the OPEN system...")
result_mesolve = run_mesolve_simulation(rho0, H, tlist, open_system_ops)

print("Running Manual simulation using the EXACT SAME Kraus map...")
# This new function uses the same Kraus map as Qiskit
rhos_manual_kraus = run_kraus_simulation(rho0, H, tlist, open_system_ops)


# --- 6. Calculate the EXACT Analytical Solution for Comparison ---
pop_qutip = 0.5 * (1 - np.array(result_mesolve.expect[2]))  # Convert <σz> to excited‐state population: P1 = (1 - <σz>)/2
rhos_qutip = [state.full() for state in result_mesolve.states]
analytical_solution_pop = (1 + np.cos(g * tlist)) / 2.0
rhos_ana = []
for t in tlist:
    U_t = expm(-1j * H * t)
    rhos_ana.append(U_t @ rho0 @ U_t.conj().T)
rhos_ana = np.array(rhos_ana)

# --- 7. Define Uhlmann fidelity function ---
def fidelity(rho, sigma):
    # Ensure matrices are numpy arrays
    rho = np.asarray(rho)
    sigma = np.asarray(sigma)
    sqrt_rho = sqrtm(rho)
    inner = sqrtm(sqrt_rho @ sigma @ sqrt_rho)
    return np.real(np.trace(inner)**2)

# --- 8. Compute fidelity over time ---
fids_closed_sim_vs_ana = [fidelity(rhos_ana[i], rhos_closed[i]) for i in range(len(tlist))]
fids_hw_vs_ana = [fidelity(rhos_ana[i], rhos_hw[i]) for i in range(len(tlist))]

print("Calculating fidelity of the open system vs. analytical truth...")
fids_open_vs_ana = [fidelity(rhos_ana[i], rhos_strang[i]) for i in range(len(tlist))]
fids_qutip_vs_ana = [fidelity(rhos_ana[i], rhos_qutip[i]) for i in range(len(tlist))]

# Compare our Manual OPEN simulation directly to the trusted QuTiP OPEN simulation.
# This should be perfectly 1.0 if our solver is correct.
print("Calculating fidelity to measure numerical accuracy of the manual solver...")
fids_open_vs_qutip = [fidelity(rhos_hw[i], rhos_qutip[i]) for i in range(len(tlist))]
fids_open_vs_strang = [fidelity(rhos_hw[i], rhos_strang[i]) for i in range(len(tlist))]
fids_open_vs_kraus = [fidelity(rhos_hw[i], rhos_manual_kraus[i]) for i in range(len(tlist))]


# --- 9. Plot Populations Results ---
print("Plotting the results...")
plt.figure(figsize=(10, 6))
plt.plot(tlist, pop_closed_stable, 'b--', lw=2, label='Closed System (Manual Sim)')
plt.plot(tlist, pop_open_stable, 'r-', lw=2, label='Open System (Manual Sim)')
plt.plot(tlist, analytical_solution_pop, 'k:', lw=2, label='Exact Analytical Solution')
plt.title("Manual Simulation of Qubit Evolution")
plt.xlabel("Time")
plt.ylabel("Population in Excited State |1>")
plt.ylim(-0.1, 1.1)
plt.legend()
plt.grid(True)
plt.show()


# --- 10. Plot fidelity alongside populations ---
plt.figure(figsize=(12, 5))

# Plot populations from the qiskit vs analytical solution
plt.subplot(1, 2, 1)
plt.plot(tlist, np.real([r[1, 1] for r in rhos_ana]), 'k:', lw=3, label='Analytical Pop.')
plt.plot(tlist, np.real([r[1, 1] for r in rhos_hw]), 'r-', label='Qiskit Sim Pop.')
plt.plot(tlist, pop_qutip,   'b-',  label='QuTiP mesolve')
plt.xlabel('Time'); plt.ylabel('P₁'); plt.legend(); plt.title('Populations: Qiskit vs Qutip vs Analytical')
plt.grid(True)

# Plot fidelity of both our manual simulation and the qiskit data vs the analytical truth
plt.subplot(1, 2, 2)
#plt.plot(tlist, fids_closed_sim_vs_ana, 'g-', lw=3, label='Fidelity(Manual CLOSED Sim, Analytical)')
#plt.plot(tlist, fids_hw_vs_ana, 'm--', lw=2, label='Fidelity(Qiskit CLOSED Sim, Analytical)')
#plt.plot(tlist, fids_open_vs_ana, 'r-', lw=3, label='Fidelity(Manual OPEN Sim, Analytical)')
#plt.plot(tlist, fids_qutip_vs_ana, 'p-', lw=3, label='Fidelity(QUTIP OPEN Sim, Analytical)')
plt.plot(tlist, fids_open_vs_strang, 'brown', linestyle=':', lw=4, label='Numerical Accuracy (Simulated vs. Strang)')
plt.plot(tlist, fids_open_vs_qutip, 'black', linestyle=':', lw=4, label='Numerical Accuracy (Simulated vs. QuTiP)')
plt.plot(tlist, fids_open_vs_kraus, 'red', linestyle=':', lw=4, label='Numerical Accuracy (Simulated vs. Manual Kraus)')
plt.xlabel('Time'); plt.ylabel('Fidelity'); plt.ylim(-0.1, 1.1)
plt.legend(); plt.title('State Fidelity vs. Analytical Truth')
plt.grid(True)

plt.tight_layout()
plt.show()


# Extract the expectation values. result.expect is now a list of 3 arrays.
expect_x = result_mesolve.expect[0]
expect_y = result_mesolve.expect[1]
expect_z = result_mesolve.expect[2]

print("Plotting expectation values from QuTiP mesolve...")
plt.figure(figsize=(10, 6))

plt.plot(tlist, expect_x, 'b-', lw=2, label=r'$\langle\sigma_x\rangle$')
plt.plot(tlist, expect_y, 'g-', lw=2, label=r'$\langle\sigma_y\rangle$')
plt.plot(tlist, expect_z, 'r-', lw=2, label=r'$\langle\sigma_z\rangle$')

plt.title("Expectation Values from QuTiP mesolve (Open System)")
plt.xlabel("Time")
plt.ylabel("Expectation Value")
plt.legend()
plt.grid(True)
plt.show()


if USE_MULTIPLE_QUBITS:

    # --- 9. Run Simulation and Plot N-Qubit Thermalization ---
    # NOTE: The Qiskit and Analytical sections are no longer valid for this N-qubit system.
    # We will only run our manual open system simulation.
    print(f"Running simulation for the {N}-Qubit Thermal Chain...")
    #pop_open_stable, rhos_strang = run_stable_simulation(rho0, H, tlist, open_system_ops)

    # --- Calculate Expectation Values for Each Qubit ---
    print("Calculating local expectation values...")
    expect_z = np.zeros((N, len(tlist)))
    for k in range(N):
        sz_k = promote_op(sigmaz, k, N)
        for t_idx, rho_t in enumerate(rhos_strang):
            expect_z[k, t_idx] = np.real(np.trace(rho_t @ sz_k))

    # --- Plot the Results ---
    print("Plotting the results...")
    plt.figure(figsize=(12, 7))
    for k in range(N):
        # We plot (1 - <σz>)/2 to get the population of the excited state Pk(1)
        # This is a good proxy for local temperature.
        population_k = (1 - expect_z[k, :]) / 2
        plt.plot(tlist, population_k, label=f'Qubit {k+1} Excitation')

    plt.title(f"Thermalization of an {N}-Qubit Chain")
    plt.xlabel("Time")
    plt.ylabel("Population in Excited State (Proxy for Temperature)")
    plt.legend()
    plt.grid(True)
    plt.show()
