import os

import experiment

e = experiment.Experiment("tests/test_experiment.json")
e.save_to_json("results.json")

with open("results.json") as f:
    print(f.read(), end="")

os.remove("results.json")