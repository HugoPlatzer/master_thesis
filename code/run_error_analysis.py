import sys
import json
import csv
from pathlib import Path

import samplers
from tokenizer import ASCIITokenizer
from dataset import create_dataset
from model import load_model_from_path, generate_responses


if len(sys.argv) != 2:
    print(f"usage: {sys.argv[0]} error_analysis.json")
    exit(1)

config_file = Path(sys.argv[1])
config = json.loads(open(config_file).read())

sampler_name = config["sampler"]["name"]
sampler_class = getattr(samplers, sampler_name)
sampler_params = config["sampler"]["params"]
sampler = sampler_class(**sampler_params)

model_path = config["model"]
model = load_model_from_path(model_path)

tokenizer = ASCIITokenizer()

dataset_size = config["dataset_size"]
dataset = create_dataset(sampler, tokenizer, dataset_size)
prompts = dataset["prompt_str"]
correct_responses = dataset["response_str"]

batch_size = config["batch_size"]
model_responses = generate_responses(
        model, tokenizer, prompts, batch_size)

wrong_responses = [(prompt, model_resp, correct_resp)
        for prompt, model_resp, correct_resp
        in zip(prompts, model_responses, correct_responses)
        if model_resp != correct_resp]
num_wrong = len(wrong_responses)
num_total = len(prompts)

output_file_path = config_file.with_suffix(".csv")
with open(output_file_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["prompt", "model_resp", "correct_resp"])
    writer.writerows(wrong_responses)

print(f"{num_wrong}/{num_total} wrong responses")
print(f"wrong responses written to {output_file_path}")
