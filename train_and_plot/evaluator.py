import torch

class Evaluator:
    def __init__(self, sampler, model, num_samples, strip_scratchpad):
        self.sampler = sampler
        self.model = model
        self.num_samples = num_samples
        self.strip_scratchpad = strip_scratchpad
    
    def get_params(self):
        return {
            "num_samples" : self.num_samples,
            "strip_scratchpad": self.strip_scratchpad
        }
    
    def __str__(self):
        params = {
            "sampler": self.sampler,
            "model": self.model
        }
        params.update(self.get_params())
        params_str = ", ".join(
            f"{name}={value}" for name, value in params.items())
        return f"{self.__class__.__name__}({params_str}"")"
    
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
