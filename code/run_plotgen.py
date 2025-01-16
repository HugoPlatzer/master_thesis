import sys
import json

import plotgen


if len(sys.argv) != 2:
    print(f"usage: {sys.argv[0]} plot.json")
    exit(1)

plot_config_file = sys.argv[1]
plot_config = json.loads(open(plot_config_file).read())

plot_type = plot_config["plot_type"]
plot_generator_name = f"plot_{plot_type}"
plot_generator = getattr(plotgen, plot_generator_name)

plot_generator.create_plot(plot_config_file)