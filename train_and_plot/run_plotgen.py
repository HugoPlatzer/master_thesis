import sys

import plot_generator

if len(sys.argv) != 3:
    print(f"usage: {sys.argv[0]} EXPERIMENT_RESULTS_FILE PLOT_FILE")
    exit(1)

results_file = sys.argv[1]
plot_file = sys.argv[2]

plotgen = plot_generator.PlotGenerator(results_file)
plotgen.generate_plot(plot_file)