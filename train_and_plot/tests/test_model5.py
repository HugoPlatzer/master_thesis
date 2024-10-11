import model

import sys

model_filename = sys.argv[1]
prompt = sys.argv[2]

m = model.GPT2Model.load_from_file(model_filename)
answer = m.answer_prompts([prompt])[0]
print(repr(answer))