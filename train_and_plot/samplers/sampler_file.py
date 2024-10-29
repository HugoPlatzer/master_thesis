import random

from .sampler import Sampler

class SamplerFile(Sampler):
    def __init__(self, file_name):
        super().__init__(file_name=file_name)
        self.load_file()
    
    def load_file(self):
        file_name = self.params["file_name"]
        self.prompts_responses = []
        for line in open(file_name).readlines():
            line_parts = line.strip().split("=")
            if len(line_parts) != 2:
                raise Exception(f"invalid line in file: {repr(line)}")
            prompt = line_parts[0] + "="
            response = line_parts[1]
            self.prompts_responses.append((prompt, response))
    
    def get_prompt_and_response(self):
        return random.choice(self.prompts_responses)
    
    def get_max_prompt_len(self):
        return max(len(prompt) for prompt, response
            in self.prompts_responses)
    
    def get_max_response_len(self):
        return max(len(response) for prompt, response
            in self.prompts_responses)
