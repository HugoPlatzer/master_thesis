import os

import experiment

e = experiment.Experiment("tests/test_experiment.json")
e.run_training()
e.save_results("results.json")

with open("results.json") as f:
    print(f.read(), end="")

os.remove("results.json")
