import json
import os
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import numpy as np

from .settings import apply_plot_settings

def create_plot(config_file):
    apply_plot_settings()
    config = json.loads(open(config_file).read())
    
    curves = []
    for curve_params in config["curves"]:
        results_data = json.loads(
            open(curve_params["results_file"]).read())
        curve_data = [(state["step"], state["val_accuracy"])
            for state in results_data["training_metrics"]
            if "val_accuracy" in state]
        curve_colormap = plt.get_cmap(curve_params["colormap"])
        curve_color = curve_colormap(
            curve_params["colormap_position"])

        curve = {
            "name": curve_params["name"],
            "data": curve_data,
            "color": curve_color
        }
        curves.append(curve)
    
    for i, curve in enumerate(curves):
        x_values = [d[0] for d in curve["data"]]
        y_values = [d[1] for d in curve["data"]]
        line, = plt.plot(x_values, y_values,
            color=curve["color"])
        plt.plot(x_values[-1], y_values[-1], marker="s", 
            color=curve["color"],
            markersize=10,
            alpha=0.5,
            label=curve["name"])
    
    plt.xscale("log")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Training step")
    plt.ylabel("Accuracy")
    
    plt.grid(True)
    plt.legend(loc="lower right", bbox_to_anchor=(1,1),
        ncol=config["legend_columns"])
    plt.tight_layout()
    output_file = config_file.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(output_file)
