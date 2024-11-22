import json
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MultipleLocator, FixedFormatter
import numpy as np
import csv

def create_csv(config_file, col_labels, curves):
    csv_file = config_file.rsplit(".", 1)[0] + ".csv"
    csv_rows = []
    csv_rows.append(["name"] + col_labels)
    
    for curve in curves:
        y_values_formatted = [f"{y:.3f}" for y in curve["y_values"]]
        row = [curve["name"]] + y_values_formatted
        csv_rows.append(row)
    
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)


def create_plot(config_file):
    config = json.loads(open(config_file).read())
    
    x_tick_labels = config["x_axis_points"]
    x_values = list(range(len(x_tick_labels)))
    
    curves = []
    for curve in config["curves"]:
        y_values = []
        for results_file in curve["results_files"]:
            results_data = json.loads(open(results_file).read())
            test_accuracy = results_data["best_model"]["test_accuracy"]
            y_values.append(test_accuracy)
        
        curve_data = {
            "name": curve["name"],
            "y_values": y_values
        }
        curves.append(curve_data)
    
    fig, ax = plt.subplots()
    
    colors = plt.colormaps["tab10"].colors
    
    for i, curve in enumerate(curves):
        ax.plot(x_values, curve["y_values"], label=curve["name"],
            marker="s", color=colors[i])
    
    ax.xaxis.set_major_locator(FixedLocator(x_values))
    ax.xaxis.set_major_formatter(FixedFormatter(x_tick_labels))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    plt.ylim(-0.05, 1.05)
    
    plt.xlabel(config["x_axis_label"])
    plt.ylabel("Accuracy")
    
    plt.grid(True)
    plt.legend(loc="lower right", bbox_to_anchor=(1,1))
    plt.tight_layout()
    output_file = config_file.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(output_file)
    
    create_csv(config_file, x_tick_labels, curves)