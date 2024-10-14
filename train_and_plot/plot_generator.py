import json
import matplotlib.pyplot as plt

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
        
        fig, ax1 = plt.subplots()
        ax1.plot(x_values, y_values_loss, color="blue", label="Loss")
        ax1.set_xlim(0, max(x_values) * 1.05)
        ax1.set_ylim(0, max(y_values_loss) * 1.05)
        ax1.set_xlabel("Training step")
        ax1.set_ylabel("Loss")
        
        ax2 = ax1.twinx()
        ax2.plot(x_values, y_values_acc, color="red", label="Accuracy")
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel("Accuracy")
        
        plt.tight_layout()
        plt.savefig(output_file)
