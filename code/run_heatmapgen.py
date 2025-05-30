import sys
import json
import torch
import matplotlib.pyplot as plt

from model import load_model_from_path
from tokenizer import ASCIITokenizer


if len(sys.argv) != 2:
    print(f"usage: {sys.argv[0]} config.json")
    exit(1)

config_file = sys.argv[1]
config = json.loads(open(config_file).read())

model = load_model_from_path(config["model_path"])

tokenizer = ASCIITokenizer()
token_seq = tokenizer.encode(config["prompt_str"],
        add_special_tokens=False)
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

fig, axes = plt.subplots(nrows=num_layers, ncols=num_heads,
        figsize=(12, 12))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

prompt_str = config["prompt_str"]
prompt_str_shifted = config["prompt_str"][1:] + "."

for layer_idx in range(num_layers):
    for head_idx in range(num_heads):
        M = attention_matrices[layer_idx][head_idx]
        row_index = num_layers - 1 - layer_idx
        col_index = head_idx
        ax = axes[row_index][col_index]
        ax.imshow(M)
        ax.set_xticks(range(len(prompt_str)))
        ax.set_xticklabels(prompt_str)
        ax.set_yticks(range(len(prompt_str_shifted)))
        ax.set_yticklabels(prompt_str_shifted)

for layer_idx in range(num_layers):
    row_index = num_layers - 1 - layer_idx
    axes[row_index][0].set_ylabel(f"Layer {layer_idx + 1}")

for head_idx in range(num_heads):
    axes[num_layers - 1][head_idx].set_xlabel(f"Head {head_idx + 1}")

plt.tight_layout()
output_file = config_file.rsplit(".", 1)[0] + ".pdf"
plt.savefig(output_file)
