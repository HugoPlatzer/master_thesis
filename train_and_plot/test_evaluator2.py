from sampler import SamplerStringReverse
from model import GPT2Model
from evaluator import Evaluator

import sys

# test only on full

model_filename = sys.argv[1]
strlen = int(sys.argv[2])

model = GPT2Model.load_from_file(model_filename)
sampler = SamplerStringReverse(max_len=strlen, mixed_len=False)
evaluator = Evaluator(sampler, model, num_samples=10)

print(evaluator)
accuracy = evaluator.evaluate_model(debug=True)
print(accuracy)