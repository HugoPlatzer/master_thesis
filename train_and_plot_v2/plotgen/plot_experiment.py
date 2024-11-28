import json
import os
import matplotlib.pyplot as plt

from .settings import apply_plot_settings

def create_plot(config_file):
    apply_plot_settings()
    
    config = json.loads(open(config_file).read())
    results_file = config["results_file"]
    results = json.loads(open(results_file).read())
    
    #plt.style.use("ggplot")
    
    batch_loss_data = [(state["step"], state["avg_batch_loss"])
        for state in results["training_metrics"]
        if "avg_batch_loss" in state]
    batch_loss_x = [state[0] for state in batch_loss_data]
    batch_loss_y = [state[1] for state in batch_loss_data]
    
    fig, ax1 = plt.subplots()
    ax1.plot(batch_loss_x, batch_loss_y, color="b",
        label="Batch loss")
    ax1.plot(batch_loss_x[-1], batch_loss_y[-1], marker="s", color="b")
    ax1.set_ylim(0.0, max(batch_loss_y) * 1.05)
    
    val_accuracy_data = [(state["step"], state["val_accuracy"])
        for state in results["training_metrics"]
        if "val_accuracy" in state]
    val_accuracy_x = [state[0] for state in val_accuracy_data]
    val_accuracy_y = [state[1] for state in val_accuracy_data]
    
    ax2 = ax1.twinx()
    ax2.plot(val_accuracy_x, val_accuracy_y, color="r",
        label="Validation accuracy")
    ax2.plot(val_accuracy_x[-1], val_accuracy_y[-1], marker="s", color="r")
    ax2.set_ylim(-0.05, 1.05)
    
    ax1.xaxis.grid(True)
    ax2.yaxis.grid(True)
    
    if "x_limits" in config:
        plt.xlim(config["x_limits"][0], config["x_limits"][1])
    
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy")
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="lower right", bbox_to_anchor=(1,1))
    
    plt.tight_layout()
    output_file = config_file.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(output_file)