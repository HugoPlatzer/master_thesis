import samplers
import evaluator

sampler = samplers.SamplerStringReverse(max_len=3, mixed_len=False)

class MockModel:
    def answer_prompts(self, prompts):
        responses = []
        for p in prompts:
            in_str = p[:-1]
            out_str = in_str[::-1]
            responses.append(out_str)
        return responses

class MockModelWithScratchpad:
    def answer_prompts(self, prompts):
        responses = []
        for p in prompts:
            in_str = p[:-1]
            out_str = in_str[::-1]
            scratch_part = f"[{in_str}]"
            responses.append(scratch_part + out_str)
        return responses


model1 = MockModel()
model2 = MockModelWithScratchpad()

def test_eval_noscratch_nostrip():
    ev = evaluator.Evaluator(sampler, model1, num_samples=10, strip_scratchpad=False)
    score = ev.evaluate_model(debug=True)
    print("score (scratchpad not present, not stripping):", score)
    assert score == 1.0
    
def test_eval_noscratch_strip():
    ev = evaluator.Evaluator(sampler, model1, num_samples=10, strip_scratchpad=True)
    score = ev.evaluate_model(debug=True)
    print("score (scratchpad not present, stripping):", score)
    assert score == 1.0

def test_eval_scratch_nostrip():
    ev = evaluator.Evaluator(sampler, model2, num_samples=10, strip_scratchpad=False)
    score = ev.evaluate_model(debug=True)
    print("score (scratchpad present, not stripping):", score)
    assert score == 0.0
    
def test_eval_scratch_strip():
    ev = evaluator.Evaluator(sampler, model2, num_samples=10, strip_scratchpad=True)
    score = ev.evaluate_model(debug=True)
    print("score (scratchpad present, stripping):", score)
    assert score == 1.0

