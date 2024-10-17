from samplers import SamplerStringReverse
from model import GPT2Model
from evaluator import Evaluator

def test():
    model = GPT2Model(max_prompt_len=8, max_response_len=8, n_embd=768, n_layer=12, n_head=12)
    strlen = 5
    
    sampler = SamplerStringReverse(max_len=strlen, mixed_len=False)
    evaluator = Evaluator(sampler, model, num_samples=10)
    
    print(evaluator)
    accuracy = evaluator.evaluate_model(debug=True)
    print(accuracy)
