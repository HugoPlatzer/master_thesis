import random
import string



class Sampler:
    def __init__(self, **kwargs):
        self.params = {name:value for name, value in kwargs.items()}
    
    def __str__(self):
        s = f"{self.__class__.__name__}("
        for name, value in self.params.items():
            s += f"{name}={value}, "
        if s.endswith(", "):
            s = s[:-2]
        s += ")"
        return s
    
    def get_prompt_and_response(self):
        pass
    
    def get_max_prompt_len(self):
        pass
    
    def get_max_response_len(self):
        pass


# params:
#   max_len: int, maximum length of sample string
#   mixed_len: bool, mix length from 1 to max_len or all samples have max_len
class SamplerStringReverse(Sampler):
    def random_str(self, slen):
        return "".join(
            random.choice(string.ascii_lowercase) for i in range(slen))
    
    def get_prompt_and_response(self):
        if self.params["mixed_len"]:
            slen = random.randint(1, self.params["max_len"])
        else:
            slen = self.params["max_len"]
        s = self.random_str(slen)
        prompt = f"{s}:"
        response = s[::-1]
        return prompt, response
    
    def get_max_prompt_len(self):
        return self.params["max_len"] + 1  # maximum string length plus : character

    def get_max_prompt_len(self):
        return self.params["max_len"] # maximum string length