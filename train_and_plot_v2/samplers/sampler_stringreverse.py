import random
import time
import string

class SamplerStringReverse:
    def __init__(self, **kwargs):
        self.string_length = kwargs["string_length"]
    
    def get_sample(self):
        chars = string.ascii_lowercase
        s = "".join(random.choice(chars) for i in range(self.string_length))
        prompt = s + ":"
        response = s[::-1]
        return {"prompt": prompt, "response": response}
