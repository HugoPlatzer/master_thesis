from .sampling import (
    get_sample_int,
    get_reversed_result,
    trim_scratchpad)

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


    @staticmethod
    def get_mul_scratchpad(a, b):
        def digit_product_block(multiplicand: str, d: int) -> str:
            carry = 0
            steps = []
            for m in reversed(multiplicand):  # units → most-significant
                m_int = int(m)
                prod = m_int * d + carry
                digit = prod % 10
                carry = prod // 10
                # step string order: multiplicand_digit, multiplier_digit,
                #                    carry_in, product_digit, carry_out
                steps.append(f"{m_int}{d}{(prod-digit)//10}{digit}{carry}")
            return "{" + ",".join(steps) + "}", carry
        
        a_str, b_str = str(a), str(b)
        len_b = len(b_str)
        
        rows = []  # textual rows  '*80{…}5360,' etc.
        addends = []  # the shifted numeric strings  '5360', '603', …
        digit_blocks = []  # just the {…} parts (need them while building)

        for place, ch in enumerate(b_str):  # left→right
            d = int(ch)
            shift = len_b - place - 1  # units shift (place value)
            shift_val = d * (10**shift)  # the “*80”, “*9000”, …
            block, _ = digit_product_block(a_str, d)  # text + (unused) carry

            # product of multiplicand and single digit (unshifted!)
            unshifted_prod = a * d
            shifted_prod = unshifted_prod * (10**shift)
            addends.append(str(shifted_prod))

            # comma after every row except the last one
            comma = "," if place < len_b - 1 else ""
            rows.append(f"*{shift_val}{block}{shifted_prod}{comma}")
        
        max_len = max(len(x) for x in addends) if addends else 1
        steps = []
        carry = 0
        # columns: units (idx 0) → most-significant
        for col in range(max_len):
            # gather the digits that really exist in this column
            summand_digits = []
            column_sum = carry
            for add in addends:  # keep original order
                if col < len(add):
                    digit = int(add[-1 - col])  # right-to-left indexing
                    summand_digits.append(str(digit))
                    column_sum += digit
            if not summand_digits:  # shouldn’t happen, but safer
                summand_digits.append("0")
            summand_str = "".join(summand_digits)
            digit_res = column_sum % 10
            new_carry = column_sum // 10
            steps.append(f"{summand_str},{carry},{digit_res},{new_carry}")
            carry = new_carry
        # if there is still carry left, consume it column-by-column
        while carry:
            digit_res = carry % 10
            new_carry = carry // 10
            steps.append(f"{digit_res},{carry//10},{digit_res},{new_carry}")
            carry = new_carry

        addition_block = "{" + ";".join(steps) + "}"
        
        scratchpad = (
            f"{a}*{b}=[\n"
            + "\n".join(rows)
            + "\n"
            + "|\n"
            + addition_block
            + "\n"
            + f"]{a * b}"
        )
        return scratchpad


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
        elif self.intermediate_steps == "scratchpad":
            response = SamplerMul.get_mul_scratchpad(a, b)
            response = trim_scratchpad(response)
        elif self.intermediate_steps == "none":
            response = c_str
        else:
            raise Exception("invalid intermediate steps type")

        return {"prompt": prompt, "response": response}
