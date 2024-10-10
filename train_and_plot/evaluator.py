import torch

class Evaluator:
    def __init__(self, sampler, model, num_samples):
        self.sampler = sampler
        self.model = model
        self.num_samples = num_samples
    
    def __str__(self):
        return (f"{self.__class__.__name__}("
        f"sampler={self.sampler}, "
        f"model={self.model}], "
        f"num_samples={self.num_samples})"
        )
    
    def evaluate_model(self, debug=False):
        prompts_responses = [self.sampler.get_prompt_and_response()
            for i in range(self.num_samples)]
        prompts = [a[0] for a in prompts_responses]
        correct_responses = [a[1] for a in prompts_responses]
        model_responses = self.model.answer_prompts(prompts)
        if not len(prompts) == len(correct_responses) == len(model_responses):
            raise ValueError("wrong number of responses from model")
        total_answers = len(model_responses)
        good_answers = 0
        for prompt, correct_response, model_response in zip(
            prompts, correct_responses, model_responses):
            if debug:
                print(f"prompt={repr(prompt)}, "
                    f"correct_response={repr(correct_response)}, "
                    f"model_response={repr(model_response)}")
            if model_response == correct_response:
                good_answers += 1
        accuracy = good_answers / total_answers
        return accuracy
        