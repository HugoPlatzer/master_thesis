import random

from . import scratchpad
from .sampler import Sampler

class SamplerMul(Sampler):
    def __init__(self, max_len, mixed_len, reverse_result, scratchpad_type):
        super().__init__(max_len=max_len, mixed_len=mixed_len,
            reverse_result=reverse_result, scratchpad_type=scratchpad_type)
    
    def random_int(self, n_digits):
        lower_limit = 10**(n_digits - 1)
        upper_limit = 10**n_digits - 1
        # limits are both inclusive in randint
        return random.randint(lower_limit, upper_limit)
    
    def get_number_of_operand_digits(self):
        if self.params["mixed_len"]:
            return random.randint(1, self.params["max_len"])
        else:
            return self.params["max_len"]
    
    def build_response(self, x, y):
        if self.params["scratchpad_type"] == "none":
            if self.params["reverse_result"]:
                return str(x * y)[::-1]
            else:
                return str(x * y)
        elif self.params["scratchpad_type"] == "basic":
            pad = scratchpad.build_scratchpad_mul(x, y)
            # scratchpad already contains product
            return pad
        else:
            raise ValueError("invalid scratchpad type")

    def get_prompt_and_response(self):
        digits_x = self.get_number_of_operand_digits()
        digits_y = self.get_number_of_operand_digits()
        x = self.random_int(digits_x)
        y = self.random_int(digits_y)
        prompt_str = f"{x}*{y}="
        response_str = self.build_response(x, y)
        return prompt_str, response_str
    
    def get_max_prompt_len(self):
        # length of both operands plus *= symbols
        return 2 * self.params["max_len"] + 2
    
    def get_max_response_len(self):
        # the longest possible response string is generated
        # when computing (10**max_digits - 1) * (10**max_digits - 1)
        x_for_longest_response = 10**self.params["max_len"] - 1
        response = self.build_response(x_for_longest_response,
            x_for_longest_response)
        return len(response)
