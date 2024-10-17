import os

import experiment

def test():
    e = experiment.Experiment("tests/test_experiment.json")
    e.save_results("results.json")
    
    with open("results.json") as f:
        print(f.read(), end="")
    
    os.remove("results.json")
