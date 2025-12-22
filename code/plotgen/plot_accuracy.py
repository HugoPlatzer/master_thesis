import json
import os
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import numpy as np

from . import settings


def create_plot(config_file):
    settings.apply_font_settings()
    config = json.loads(open(config_file).read())

    LEGEND_GAP_INCHES = 0.05

    axes_inches = settings.PLOT_GRID_SIZE
    # to ensure proper axes size in the final plot:
    # create a figure with sufficient size to fit everything
    # since axes are initialized with dimensions relative to the size
    # of the total figure, this leads to correctly sized axes
    # with sufficient excess around for later cropping
    # using bbox_inches="tight"
    excess_factor = 5
    figsize = (axes_inches[0] * excess_factor,
            axes_inches[1] * excess_factor)
    axes_dimensions = (0.4, 0.4, 0.2, 0.2)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(axes_dimensions)
    
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
        line, = ax.plot(x_values, y_values,
            color=curve["color"])
        ax.plot(x_values[-1], y_values[-1], marker="s", 
            color=curve["color"],
            markersize=10,
            alpha=0.5,
            label=curve["name"])
    
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Accuracy")
    
    ax.grid(True)
    ax.minorticks_on()
    ax.grid(which="major", linestyle="-", linewidth=0.5, color="black")
    ax.grid(which="minor", linestyle="-", linewidth=0.5, color="lightgray")
    
    axes_height = figsize[1] / excess_factor
    legend_gap_factor = LEGEND_GAP_INCHES / axes_height 

    # legend centering on top of axes:
    # anchor of legend is at its lower center
    # place legend so this anchor is at the top center of the axes
    ax.legend(loc="lower center",
            bbox_to_anchor=(0.5, 1.0 + legend_gap_factor),
        ncol=config["legend_columns"])
    output_file = config_file.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(output_file, bbox_inches="tight")
