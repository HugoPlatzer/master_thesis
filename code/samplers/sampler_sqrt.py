import math
from .sampling import (
    get_sample_int,
    get_reversed_result,
    trim_scratchpad)

class SamplerSqrt:
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

    
    @staticmethod
    def get_sqrt_scratchpad(a):
        scratchpad = f"["

        low, high = 0, a
        steps = []

        while high - low > 1:
            mid = (low + high) // 2  # floor division
            square = mid * mid
            steps.append(f"{{{low},{high}}}{mid}*{mid}={square}|")

            if square <= a:
                low = mid
            else:
                high = mid

        # After loop, check which one is the correct integer square root
        if high * high <= a:
            result = high
        else:
            result = low

        # Add the final step
        steps.append(f"{{{low},{high}}}")

        # Close the scratchpad
        scratchpad += "\n  " + "\n  ".join(steps)
        scratchpad += "\n]{}".format(result)

        return scratchpad


    def get_sample(self):
        a = get_sample_int(self.digits, self.sampling_strategy)
        a = max(1, a)
        k = math.isqrt(a)
        a_str = str(a).zfill(self.digits)
        max_response_len = math.ceil(self.digits / 2)
        k_str = str(k).zfill(max_response_len)
        prompt = f"{a_str}:"

        if self.intermediate_steps == "reverse":
            response = get_reversed_result(k_str) + k_str
        elif self.intermediate_steps == "scratchpad":
            response = SamplerSqrt.get_sqrt_scratchpad(a)
            response = trim_scratchpad(response)
        elif self.intermediate_steps == "none":
            response = k_str
        else:
            raise Exception("invalid intermediate steps type")

        return {"prompt": prompt, "response": response}
