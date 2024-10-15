import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
    
    def generate_plot(self, output_file):
        x_values = [ts["training_step"]
            for ts in self.json_data["training_states"]]
        y_values_loss = [ts["loss"]
            for ts in self.json_data["training_states"]]
        y_values_acc = [ts["accuracy"]
            for ts in self.json_data["training_states"]]
        
        plt.style.use("ggplot")
        loss_color = (plt.rcParams['axes.prop_cycle'].by_key()
            ['color'][1])
        accuracy_color = (plt.rcParams['axes.prop_cycle'].by_key()
            ['color'][0])
        
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 1)
        
        ax1 = fig.add_subplot(gs[0, 0], zorder=1)
        ax1.set_xlim(0, max(x_values) * 1.05)
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True)
        ax1.tick_params(bottom=False, labelbottom=False,
            left=False, labelleft=False)
        
        ax2 = fig.add_subplot(gs[0, 0], zorder=2, frame_on=False)
        ax2.plot(x_values, y_values_loss, label="Loss", color=loss_color)
        ax2.set_xlim(0, max(x_values) * 1.05)
        ax2.set_ylim(0, max(y_values_loss) * 1.05)
        ax2.set_xlabel("Training step")
        ax2.set_ylabel("Loss")
        ax2.grid(False)
        
        ax3 = ax2.twinx()
        ax3.plot(x_values, y_values_acc, color="red", label="Accuracy")
        ax3.set_xlim(0, max(x_values) * 1.05)
        ax3.set_ylim(-0.05, 1.05)
        ax3.set_ylabel("Accuracy")
        ax3.grid(False)
        
        plt.tight_layout()
        plt.savefig(output_file)
