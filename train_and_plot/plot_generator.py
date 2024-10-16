import json
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (StrMethodFormatter,
    MultipleLocator, AutoMinorLocator)

class PlotGenerator:
    def __init__(self, json_file):
        self.json_file = json_file
        self.json_data = json.load(open(json_file))

    def get_params(self):
        return {
            "json_file": self.json_file
        }
    
    def __str__(self):
        params = self.get_params()
        params_str = ", ".join(
            f"{name}={value}" for name, value in params.items())
        return f"{self.__class__.__name__}({params_str}"")"
    
    @staticmethod
    def extend_limits(lower, upper, margin):
        diff = upper - lower
        return (lower - margin * diff, upper + margin * diff)
    
    def generate_plot(self, output_file):
        x_values = [ts["training_step"]
            for ts in self.json_data["training_states"]]
        y_values_loss = [ts["loss"]
            for ts in self.json_data["training_states"]]
        y_values_acc = [ts["accuracy"]
            for ts in self.json_data["training_states"]]
            
        margin_x, margin_y = 0.02, 0.03
        x_limits = self.extend_limits(0, max(x_values), margin_x)
        y_limits_loss = self.extend_limits(
            0, math.ceil(max(y_values_loss)), margin_y)
        y_limits_acc = self.extend_limits(0, 1, margin_y)
        
        minorticks_x = 4
        tick_interval_y_loss = 1.0
        minorticks_y_loss = 4
        tick_interval_y_acc = 0.2
        minorticks_y_acc = 4
        
        plt.style.use("ggplot")
        loss_color = (plt.rcParams['axes.prop_cycle'].by_key()
            ['color'][1])
        accuracy_color = (plt.rcParams['axes.prop_cycle'].by_key()
            ['color'][0])
        
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 1)
        
        ax1 = fig.add_subplot(gs[0, 0], zorder=1)
        ax1.set_xlim(*x_limits)
        ax1.set_ylim(*y_limits_acc)
        ax1.grid(which="major", visible="True", linewidth=0.7)
        ax1.grid(which="minor", visible="True", linewidth=0.2)
        ax1.xaxis.set_minor_locator(AutoMinorLocator(minorticks_x))
        ax1.yaxis.set_major_locator(MultipleLocator(tick_interval_y_acc))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(minorticks_y_acc))
        ax1.tick_params(bottom=False, labelbottom=False,
            left=False, labelleft=False)

        
        ax2 = fig.add_subplot(gs[0, 0], zorder=2, frame_on=False)
        ax2.plot(x_values, y_values_loss, label="Loss", color=loss_color)
        ax2.set_xlim(*x_limits)
        ax2.set_ylim(*y_limits_loss)
        ax2.set_xlabel("Training step")
        ax2.set_ylabel("Loss", color=loss_color)
        ax2.grid(False)
        ax2.xaxis.set_minor_locator(AutoMinorLocator(minorticks_x))
        ax2.yaxis.set_major_locator(MultipleLocator(tick_interval_y_loss))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(minorticks_y_loss))
        ax2.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        ax2.yaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
        
        
        ax3 = ax2.twinx()
        ax3.plot(x_values, y_values_acc, color="red", label="Accuracy")
        ax3.set_xlim(*x_limits)
        ax3.set_ylim(*y_limits_acc)
        ax3.set_ylabel("Accuracy", color=accuracy_color)
        ax3.grid(False)
        ax3.yaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
        ax3.yaxis.set_major_locator(MultipleLocator(tick_interval_y_acc))
        ax3.yaxis.set_minor_locator(AutoMinorLocator(minorticks_y_acc))
        
        plt.tight_layout()
        plt.savefig(output_file)
