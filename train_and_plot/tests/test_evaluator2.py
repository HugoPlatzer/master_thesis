from sampler import SamplerStringReverse
from model import GPT2Model
from evaluator import Evaluator

def run_test(model_filename, strlen):
    # test args are all passed as str
    strlen = int(strlen)
    
    model = GPT2Model.load_from_file(model_filename)
    sampler = SamplerStringReverse(max_len=strlen, mixed_len=False)
    evaluator = Evaluator(sampler, model, num_samples=10)
    
    print(evaluator)
    accuracy = evaluator.evaluate_model(debug=True)
    print(accuracy)