import argparse

from ibm_quantum_runner import QuantumRunner

parser = argparse.ArgumentParser()
parser.add_argument("execution_id")
args = parser.parse_args()

execution = QuantumRunner.from_env(dotenv_path=".env").resume(args.execution_id)
execution.wait()
print(execution.result().to_dict())
