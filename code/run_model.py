import sys

from model import load_model_from_path, generate_responses
from tokenizer import ASCIITokenizer

if len(sys.argv) != 3:
    print(f"usage: {sys.argv[0]} MODEL_PATH PROMPT_STR")
    exit(1)

model_path = sys.argv[1]
model = load_model_from_path(model_path)

prompt_str = sys.argv[2]

tokenizer = ASCIITokenizer()

model_response = generate_responses(model, tokenizer, [prompt_str], 1)
response_str = model_response[0]

print(response_str)
