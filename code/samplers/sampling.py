import random
import math

def get_sample_int(digits, sampling_strategy):
    if sampling_strategy == "basic":
        lower_limit = 10**(digits-1)
        upper_limit = 10**digits - 1
    elif sampling_strategy == "from_zero":
        lower_limit = 0
        upper_limit = 10**digits - 1
    elif sampling_strategy == "uniform_digits":
        num_digits = random.randint(1, digits)
        lower_limit = 10**(num_digits-1)
        if lower_limit == 1:
            lower_limit = 0
        upper_limit = 10**num_digits - 1
    elif sampling_strategy == "uniform_bits":
        max_num_bits = math.ceil(math.log2(10**digits))
        num_bits = random.randint(1, max_num_bits)
        lower_limit = 2**(num_bits-1)
        if lower_limit == 1:
            lower_limit = 0
        upper_limit = 2**num_bits - 1
        if upper_limit > 10**digits - 1:
            upper_limit = 10**digits - 1
    else:
        raise Exception("invalid sampling strategy")
    return random.randint(lower_limit, upper_limit)
