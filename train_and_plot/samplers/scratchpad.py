def generate_addition_scratchpad(numbers, debug=False):
    numbers = [str(x) for x in numbers]
    n_steps = max(len(x) for x in numbers)
    numbers_rev = [x[::-1] for x in numbers]
    steps_pads = []
    carry = 0
    for step in range(n_steps):
        digits_in = [int(x[step]) if step < len(x) else 0 for x in numbers_rev]
        digit_sum = (sum(digits_in) + carry) % 10
        carry_new = (sum(digits_in) + carry) // 10
        if debug:
            print(step, digits_in, carry, digit_sum, carry_new)
        digits_in_str = "".join(str(d) for d in digits_in)
        pad = f"{digits_in_str}|{carry}|{digit_sum}|{carry_new}"
        steps_pads.append(pad)
        carry = carry_new
    return ",".join(steps_pads)