import random
import time
import string

class SamplerMul:
    def __init__(self, **kwargs):
        self.digits = kwargs["digits"]
    
    def get_sample(self):
        lower_limit = 10**(self.digits-1)
        upper_limit = 10**self.digits - 1
        a = random.randint(lower_limit, upper_limit)
        b = random.randint(lower_limit, upper_limit)
        c = a * b
        prompt = f"{a}*{b}="
        response = str(c)
        max_response_len = 2 * self.digits
        response = response.zfill(max_response_len)
        return {"prompt": prompt, "response": response}