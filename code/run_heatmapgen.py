import sys
import json
import torch
import matplotlib
import matplotlib.pyplot as plt

from model import load_model_from_path
from tokenizer import ASCIITokenizer
from plotgen import settings

settings.apply_font_settings()

if len(sys.argv) != 2:
    print(f"usage: {sys.argv[0]} config.json")
    exit(1)

config_file = sys.argv[1]
config = json.loads(open(config_file).read())

plot_type = config["plot_type"]

tick_font_size = (config["tick_font_size"]
        if "tick_font_size" in config
        else settings.FONT_SIZE)

model = load_model_from_path(config["model_path"])

tokenizer = ASCIITokenizer()
input_str = config["prompt_str"] + config["response_str"]
token_seq = tokenizer.encode(input_str, add_special_tokens=False)
token_tensor = torch.tensor([token_seq])
state = model.forward(token_tensor, output_attentions=True)

attention_matrices = []
for layer_idx, layer_attn in enumerate(state.attentions):
    attention_matrices.append([])
    for head_idx, head_attn in enumerate(layer_attn[0]):
        attn_matrix = head_attn.tolist()
        attention_matrices[layer_idx].append(attn_matrix)

attention_matrices = torch.tensor(attention_matrices)
num_layers = attention_matrices.size()[0]
num_heads = attention_matrices.size()[1]

figure_width = config["figure_width"]
figure_height = config["figure_height"]
colormap = config["colormap"]


def format_label(x):
    if len(x) != 1:
        raise Exception(f"invalid input '{x}'")
    if x == "{" or x == "}":
        x = "\\" + x
    return f"\\texttt{x}"

row_labels = config["response_str"] + "."
col_labels = config["prompt_str"] + config["response_str"]
row_labels = [format_label(x) for x in row_labels]
col_labels = [format_label(x) for x in col_labels]


# only keep rows of attention matrices where tokens of
# response_str are generated, plus EOS token
attention_matrices = attention_matrices[:, :, -len(row_labels):, :]




def plot_single():
    plot_layer = config["plot_layer"]
    M = attention_matrices[plot_layer, :, :, :]
    M = torch.mean(M, dim=0)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    ax.imshow(M, aspect="auto", cmap=colormap)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    ax.tick_params(axis="both", which="major",
            labelsize=tick_font_size)


def plot_full():
    figure_gap = config["figure_gap"]

    fig, axes = plt.subplots(nrows=num_layers, ncols=num_heads,
            figsize=(figure_width, figure_height),
            gridspec_kw={"wspace": figure_gap, "hspace": figure_gap},
            sharex=True, sharey=True)

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            M = attention_matrices[layer_idx][head_idx]
            row_index = num_layers - 1 - layer_idx
            col_index = head_idx
            ax = axes[row_index][col_index]
            ax.imshow(M, aspect="auto", cmap=colormap)
            
            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(col_labels)
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels)
            ax.tick_params(axis="both", which="major",
                    labelsize=tick_font_size)

            if layer_idx != 0:
                ax.tick_params(bottom=False)
            if head_idx != 0:
                ax.tick_params(left=False)

    for layer_idx in range(num_layers):
        row_index = num_layers - 1 - layer_idx
        axes[row_index][0].set_ylabel(
                f"Block {layer_idx + 1}")

    for head_idx in range(num_heads):
        axes[num_layers - 1][head_idx].set_xlabel(
                f"Head {head_idx + 1}")


if plot_type == "single":
    plot_single()
elif plot_type == "full":
    plot_full()
else:
    raise Exception("invalid plot type", plot_type)


output_file = config_file.rsplit(".", 1)[0] + ".pdf"
plt.savefig(output_file, bbox_inches="tight")
