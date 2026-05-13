from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit import QuantumCircuit

num_qubits = 4 # From your extreme starvation ablation

# 1. Define the components
feature_map = ZFeatureMap(feature_dimension=num_qubits, reps=2)
ansatz = RealAmplitudes(num_qubits=num_qubits, reps=2, entanglement="linear")

# 2. Build the circuit
qc = QuantumCircuit(num_qubits)
qc.compose(feature_map, inplace=True)
qc.barrier() # Adds a nice visual line separating data encoding from the trainable ansatz
qc.compose(ansatz, inplace=True)

# 3. Draw it cleanly
# 'mpl' (Matplotlib) is highly recommended over 'latex' for generating standalone PNGs
# 'fold=-1' prevents the circuit from wrapping to a new line, keeping it strictly horizontal
qc.draw(output='mpl', style='clifford', scale=0.8, fold=-1, filename="resqnet_quantum_bottleneck.png")