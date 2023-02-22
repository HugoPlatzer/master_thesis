from transformers import GPT2LMHeadModel
import torch
import sys


def prepare_model_input(s):
    model_input_s = "(" + s.strip() + " "
    if model_input_s == "( ":
        model_input_s = "("
    return torch.tensor([[ord(c) for c in model_input_s]])

def convert_model_output(model_input, model_output):
    a = model_output[0][len(model_input[0]):]
    return "".join(chr(x) for x in a)

model_path = sys.argv[1]
model_input_s = sys.argv[2]


model = GPT2LMHeadModel.from_pretrained(model_path)
model_input = prepare_model_input(model_input_s)
model_output = model.generate(model_input, min_new_tokens=5, max_new_tokens=5, do_sample=True)
model_output_s = convert_model_output(model_input, model_output)
print(repr(model_output_s))
