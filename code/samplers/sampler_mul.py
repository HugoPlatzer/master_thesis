from .sampling import get_sample_int, get_reversed_result

class SamplerMul:
    def __init__(self, **kwargs):
        self.digits = kwargs["digits"]
        if "sampling_strategy" in kwargs:
            self.sampling_strategy = kwargs["sampling_strategy"]
        else:
            self.sampling_strategy = "basic"

        if "intermediate_steps" in kwargs:
            self.intermediate_steps = kwargs["intermediate_steps"]
        else:
            self.intermediate_steps = "none"
    

    def get_sample(self):
        a = get_sample_int(self.digits, self.sampling_strategy)
        b = get_sample_int(self.digits, self.sampling_strategy)
        c = a * b
        a_str = str(a).zfill(self.digits)
        b_str = str(b).zfill(self.digits)
        c_str = str(c).zfill(2 * self.digits)
        prompt = f"{a_str}*{b_str}="

        if self.intermediate_steps == "reverse":
            response = get_reversed_result(c_str) + c_str
        elif self.intermediate_steps == "none":
            response = c_str

        return {"prompt": prompt, "response": response}
