import numpy as np
import pennylane as qml

# Number of qubits in our small circuit
NUM_QUBITS = 8

# Total length of the quantum feature vector (NUM_QUBITS + some derived stats)
QUANTUM_FEATURE_LEN = NUM_QUBITS + 4

# Device: statevector simulator
dev = qml.device("default.qubit", wires=NUM_QUBITS)


def _preprocess_to_angles(x, num_qubits=NUM_QUBITS):
    """
    Convert an arbitrary 1D classical vector x into a stable vector of rotation angles
    in [-pi, pi] of length num_qubits.
    """

    x = np.asarray(x, dtype=float).ravel()

    if x.size == 0:
        # Fallback to zeros if something weird happens
        x = np.zeros(num_qubits, dtype=float)

    # Repeat or truncate to match num_qubits
    if x.size < num_qubits:
        reps = int(np.ceil(num_qubits / x.size))
        x = np.tile(x, reps)[:num_qubits]
    elif x.size > num_qubits:
        x = x[:num_qubits]

    # Normalize: zero mean, unit-ish variance
    x = x - x.mean()
    std = x.std()
    if std > 0:
        x = x / std

    # Clip to avoid huge angles, then map [-3, 3] -> [-pi, pi]
    x = np.clip(x, -3.0, 3.0)
    angles = x * (np.pi / 3.0)  # 3 -> pi

    return angles


@qml.qnode(dev)
def _feature_map_circuit(angles):
    """
    Simple entangling feature map circuit.
    angles: length NUM_QUBITS, already scaled to reasonable rotation values.
    """
    # Initial Hadamards to create superposition
    for i in range(NUM_QUBITS):
        qml.Hadamard(wires=i)

    # Encode classical data with RY rotations
    for i, theta in enumerate(angles):
        qml.RY(theta, wires=i)

    # Entangling layer (ring of CNOTs)
    for i in range(NUM_QUBITS):
        qml.CNOT(wires=[i, (i + 1) % NUM_QUBITS])

    # Measure Z-expectations as the raw quantum features
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


def compute_quantum_features(classical_vec):
    """
    Turn a 1D classical feature vector into a quantum feature vector.

    Steps:
      1. Normalize and map classical_vec to NUM_QUBITS rotation angles in [-pi, pi]
      2. Run the quantum feature map circuit to get NUM_QUBITS expectation values
      3. Add a few derived statistics for extra expressivity

    Returns:
        np.ndarray of shape (QUANTUM_FEATURE_LEN,)
    """
    angles = _preprocess_to_angles(classical_vec, NUM_QUBITS)
    z_expects = np.array(_feature_map_circuit(angles), dtype=float)

    # Some simple engineered stats from the quantum outputs
    mean = z_expects.mean()
    std = z_expects.std()
    l1 = np.sum(np.abs(z_expects))
    l2_sq = np.sum(z_expects ** 2)

    extra = np.array([mean, std, l1, l2_sq], dtype=float)

    return np.concatenate([z_expects, extra])
