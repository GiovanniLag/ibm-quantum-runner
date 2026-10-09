from qiskit import QuantumCircuit

from ibm_quantum_runner import EstimateRequest, QuantumRunner, RunConfig

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)

request = EstimateRequest(circuit=circuit, observables={"ZZ": 1.0})
runner = QuantumRunner.from_env(dotenv_path=".env")
execution = runner.submit_estimate(request, config=RunConfig(shots=10_000))
print("local execution:", execution.id)
print("IBM jobs:", execution.ibm_job_ids)
