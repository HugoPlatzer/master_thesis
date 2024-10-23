from samplers import scratchpad

def validate_scratchpad_add(x1, x2):
    pad = scratchpad.build_scratchpad_add([x1, x2])
    pad_sum = scratchpad.extract_sum_scratchpad_add(pad)
    print(x1, x2, x1 + x2, pad, pad_sum)
    assert pad_sum == x1 + x2

def test_scratchpad_add():
    validate_scratchpad_add(7, 8)
    validate_scratchpad_add(56, 78)
    validate_scratchpad_add(123, 456)
    validate_scratchpad_add(9999, 9999)

def validate_scratchpad_mul(x1, x2):
    pad = scratchpad.build_scratchpad_mul(x1, x2)
    pad_prod = scratchpad.extract_product_scratchpad_mul(pad)
    print(x1, x2, x1 * x2, pad, pad_prod)
    assert pad_prod == x1 * x2

def test_scratchpad_mul():
    validate_scratchpad_mul(7, 8)
    validate_scratchpad_mul(56, 78)
    validate_scratchpad_mul(123, 456)
    validate_scratchpad_mul(9999, 9999)