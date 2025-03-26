from .sampling import get_sample_int

class SamplerMul:
    def __init__(self, **kwargs):
        self.digits = kwargs["digits"]
        if "sampling_strategy" in kwargs:
            self.sampling_strategy = kwargs["sampling_strategy"]
        else:
            self.sampling_strategy = "basic"
    
    def get_sample(self):
        a = get_sample_int(self.digits, self.sampling_strategy)
        b = get_sample_int(self.digits, self.sampling_strategy)
        c = a * b
        prompt = f"{a}*{b}="
        response = str(c)
        max_response_len = 2 * self.digits
        response = response.zfill(max_response_len)
        return {"prompt": prompt, "response": response}
