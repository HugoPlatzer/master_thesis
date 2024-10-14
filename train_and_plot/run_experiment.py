import sys
import os

import experiment
import plot_generator


if len(sys.argv) != 2:
    print(f"usage: {sys.argv[0]} EXPERIMENT_PATH")
    print(f"usage: {sys.argv[0]} experiments/example_stringreverse")


experiment_path = sys.argv[1]
experiment_configfile = os.path.join(experiment_path, "config.json")
experiment_resultsfile = os.path.join(experiment_path, "results.json")
experiment_modelfile = os.path.join(experiment_path, "model.bin")
experiment_plotfile = os.path.join(experiment_path, "plot.pdf")

experiment_ = experiment.Experiment(experiment_configfile)
experiment_.run_training()
experiment_.save_results(experiment_resultsfile)

experiment_.model.save_to_file(experiment_modelfile)

plotgen = plot_generator.PlotGenerator(experiment_resultsfile)
plotgen.generate_plot(experiment_plotfile)