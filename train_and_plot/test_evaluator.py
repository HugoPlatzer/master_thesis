import sampler
import model
import evaluator

sampler = sampler.SamplerStringReverse(max_len=5, mixed_len=True)
model = model.GPT2Model(
    max_prompt_len=sampler.get_max_prompt_len(),
    max_response_len=sampler.get_max_response_len(),
    n_embd=128,
    n_layer=2,
    n_head=2,
)

e = evaluator.Evaluator(sampler, model, num_samples=10)
print(e)
accuracy = e.evaluate_model(debug=True)
print(accuracy)