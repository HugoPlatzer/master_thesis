import sys

import model

if len(sys.argv) != 3:
    print(f"usage: {sys.argv[0]} MODELFILE PROMPT")
    exit(1)

model_filename = sys.argv[1]
prompt = sys.argv[2]
m = model.GPT2Model.load_from_file(model_filename)
answer = m.answer_prompts([prompt])[0]
print(repr(answer))
