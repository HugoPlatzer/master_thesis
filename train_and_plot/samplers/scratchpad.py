def build_scratchpad_add(numbers, debug=False):
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

def extract_sum_scratchpad_add(pad):
    parts = pad.split(",")
    parts = [part.split("|") for part in parts]
    if parts[-1][3] != "0":
        sum_str = parts[-1][3]
    else:
        sum_str = ""
    for part in reversed(parts):
        sum_str += part[2]
    return int(sum_str)

def build_scratchpad_mul(x1, x2, debug=False):
    x1s, x2s = str(x1), str(x2)
    pad = "["
    parts = []
    for i, d2 in enumerate(x2s):
        factor = int(d2) * 10**(len(x2s) - 1 - i)
        pad += f"*{factor}="
        digit_pad = build_scratchpad_mul_digit(x1, factor, debug)
        digit_product = digit_pad.split("}")[-1]
        parts.append(digit_product)
        pad += digit_pad + ";"
    pad += "]"
    finaladd_pad = build_scratchpad_add(parts, debug)
    pad += "[" + finaladd_pad + "]"
    result = extract_sum_scratchpad_add(finaladd_pad)
    pad += str(result)
    if debug:
        print(parts)
        print(pad)
        print(result)
    if x1 * x2 != result:
        raise Exception("invalid mul", result, x1*x2)
    return pad

def build_scratchpad_mul_digit(x1, factor, debug=False):
    k = int(str(factor)[0])
    if debug:
        print("mul_digit", x1, factor, k)
    c = 0
    scr = "{"
    zeros = "0" * (len(str(factor)) - 1)
    out = zeros
    for d1s in str(x1)[::-1]:
        d1 = int(d1s)
        o = (d1 * k + c) % 10
        cp = c
        c = (d1 * k + c) // 10
        if scr[-1].isdigit():
            scr += ","
        scr += str(d1) + str(cp) + str(o) + str(c)
        out += str(o)
    out += str(c)
    out = out[::-1]
    while len(out) > 1 and out[0] == "0":
        out = out[1:]
    scr += "}" + out
    if debug:
        print(scr)
    return scr

def extract_product_scratchpad_mul(pad):
    return int(pad.split("]")[-1])