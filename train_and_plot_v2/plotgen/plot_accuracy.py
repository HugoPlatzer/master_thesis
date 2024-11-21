import json
import os
import matplotlib.pyplot as plt
import numpy as np

def create_plot(config_file):
    config = json.loads(open(config_file).read())
    
    curves = []
    for curve_params in config["curves"]:
        results_data = json.loads(open(curve_params["results_file"]).read())
        curve_data = [(state["step"], state["val_accuracy"])
            for state in results_data["training_metrics"]
            if "val_accuracy" in state]
        
        curve = {
            "name": curve_params["name"],
            "data": curve_data
        }
        curves.append(curve)
    
    colormap = plt.get_cmap("Reds")
    if len(curves) == 1:
        colors = colormap([0.5])
    else:
        colors = colormap(np.linspace(0.25, 0.75, len(curves)))
    
    for i, curve in enumerate(curves):
        x_values = [d[0] for d in curve["data"]]
        y_values = [d[1] for d in curve["data"]]
        plt.plot(x_values, y_values, label=curve["name"], color=colors[i])
        plt.plot(x_values[-1], y_values[-1], marker="s", color=colors[i])
    
    if "x_limits" in config:
        plt.xlim(config["x_limits"][0], config["x_limits"][1])
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Training step")
    plt.ylabel("Accuracy")
    
    plt.grid(True)
    plt.legend(loc="lower right", bbox_to_anchor=(1,1))
    plt.tight_layout()
    output_file = config_file.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(output_file)