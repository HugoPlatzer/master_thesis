import math
from .sampling import get_sample_int

class SamplerSqrt:
    def __init__(self, **kwargs):
        self.digits = kwargs["digits"]
        if "sampling_strategy" in kwargs:
            self.sampling_strategy = kwargs["sampling_strategy"]
        else:
            self.sampling_strategy = "basic"

    def get_sample(self):
        a = get_sample_int(self.digits, self.sampling_strategy)
        a = max(1, a)
        k = math.isqrt(a)
        a_str = str(a).zfill(self.digits)
        max_response_len = math.ceil(self.digits / 2)
        k_str = str(k).zfill(max_response_len)
        prompt = f"{a_str}:"
        response = k_str
        return {"prompt": prompt, "response": response}
