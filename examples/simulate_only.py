from qiskit import QuantumCircuit

from ibm_quantum_runner import QuantumRunner, RunConfig

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

runner = QuantumRunner(state_dir=".quantum-runs")
result = runner.simulate(circuit, config=RunConfig(shots=1_000))
print(result.counts)
