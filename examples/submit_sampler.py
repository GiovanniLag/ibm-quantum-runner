from qiskit import QuantumCircuit

from ibm_quantum_runner import QuantumRunner, RunConfig

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

runner = QuantumRunner.from_env(dotenv_path=".env")
execution = runner.submit(circuit, config=RunConfig(shots=4_000))
print("local execution:", execution.id)
print("IBM jobs:", execution.ibm_job_ids)
