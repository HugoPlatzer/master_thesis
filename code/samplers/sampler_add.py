from .sampling import (
    get_sample_int,
    get_reversed_result,
    trim_scratchpad)

class SamplerAdd:
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
    def get_add_scratchpad(a, b):
        a_str = str(a)
        b_str = str(b)
        a_rev = a_str[::-1]
        b_rev = b_str[::-1]
        
        max_len = max(len(a_rev), len(b_rev))
        a_rev = a_rev.ljust(max_len, '0')
        b_rev = b_rev.ljust(max_len, '0')
        
        carry = 0
        steps = []
        
        for i in range(max_len):
            digit_a = int(a_rev[i])
            digit_b = int(b_rev[i])
            total = digit_a + digit_b + carry
            sum_digit = total % 10
            new_carry = total // 10
            
            step_str = f"{digit_a}{digit_b}{carry}{sum_digit}{new_carry}"
            steps.append(step_str)
            
            carry = new_carry
        
        result = a + b
        return f"[{'|'.join(steps)}]{result}"


    def get_sample(self):
        a = get_sample_int(self.digits, self.sampling_strategy)
        b = get_sample_int(self.digits, self.sampling_strategy)
        c = a + b
        a_str = str(a).zfill(self.digits)
        b_str = str(b).zfill(self.digits)
        c_str = str(c).zfill(self.digits + 1)
        prompt = f"{a_str}+{b_str}="

        if self.intermediate_steps == "reverse":
            response = get_reversed_result(c_str) + c_str
        elif self.intermediate_steps == "scratchpad":
            response = SamplerAdd.get_add_scratchpad(a, b)
            response = trim_scratchpad(response)
        elif self.intermediate_steps == "none":
            response = c_str
        else:
            raise Exception("invalid intermediate steps type")
        
        return {"prompt": prompt, "response": response}
