import samplers
import model
import evaluator

def test():
    sampler = samplers.SamplerStringReverse(max_len=5, mixed_len=True)
    model_ = model.GPT2Model(
        max_prompt_len=sampler.get_max_prompt_len(),
        max_response_len=sampler.get_max_response_len(),
        n_embd=128,
        n_layer=2,
        n_head=2,
    )
    
    eval_ = evaluator.Evaluator(sampler, model_, num_samples=10)
    print(eval_)
    accuracy = eval_.evaluate_model(debug=True)
    print(accuracy)
