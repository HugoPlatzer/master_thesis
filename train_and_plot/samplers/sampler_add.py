import random

from .sampler import Sampler

class SamplerAdd(Sampler):
    def __init__(self, max_len, mixed_len, reverse_result):
        super().__init__(max_len=max_len, mixed_len=mixed_len,
            reverse_result=reverse_result)
    
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
    
    def get_prompt_and_response(self):
        digits_x = self.get_number_of_operand_digits()
        digits_y = self.get_number_of_operand_digits()
        x = self.random_int(digits_x)
        y = self.random_int(digits_y)
        prompt_str = f"{x}+{y}="
        if self.params["reverse_result"]:
            response_str = str(x + y)[::-1]
        else:
            response_str = str(x + y)
        return prompt_str, response_str
    
    def get_max_prompt_len(self):
        # length of both operands plus += symbols
        return 2 * self.params["max_len"] + 2
    
    def get_max_response_len(self):
        # max length of sum
        return self.params["max_len"] + 1
