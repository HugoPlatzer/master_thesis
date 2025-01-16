import random
import math

class SamplerSqrt:
    def __init__(self, **kwargs):
        self.digits = kwargs["digits"]

    def get_sample(self):
        lower_limit = 10**(self.digits-1)
        upper_limit = 10**self.digits - 1
        a = random.randint(lower_limit, upper_limit)
        k = math.isqrt(a)
        prompt = f"{a}:"
        response = str(k)
        max_response_len = math.ceil(self.digits / 2)
        response = response.zfill(max_response_len)
        return {"prompt": prompt, "response": response}
