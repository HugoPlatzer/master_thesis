import sys

from experiment import Experiment

if len(sys.argv) != 2:
    print(f"usage: {sys.argv[0]} EXPERIMENT_DIRECTORY")
    exit(1)

experiment_directory = sys.argv[1]
experiment = Experiment(experiment_directory)

experiment.run()
