import re

import samplers

def test_sampler_string_reverse():
    s = samplers.SamplerStringReverse(max_len=5, mixed_len=True)
    print(s)
        
    for i in range(5):
        print(s.get_prompt_and_response())

def validate_addition(prompt_str, response_str, reverse_result=False):
    m = re.match(r"(\d+)\+(\d+)=", prompt_str)
    x, y = int(m.group(1)), int(m.group(2))
    m = re.match(r"(?:\[([^\]]*)\])?(\d+)", response_str)
    if reverse_result:
        response_sum = int(m.group(2)[::-1])
    else:
        response_sum = int(m.group(2))
    assert (x + y) == response_sum
    if m.group(1) is not None: # scratchpad present in response
        scratchpad = m.group(1)
        scratch_parts = scratchpad.split(",")
        scratch_parts = [part.split("|") for part in scratch_parts]
        if scratch_parts[-1][3] != "0":
            scratch_sum = scratch_parts[-1][3]
        else:
            scratch_sum = ""
        for part in reversed(scratch_parts):
            scratch_sum += part[2]
        assert int(scratch_sum) == (x+y)

def test_sampler_add():
    s = samplers.SamplerAdd(max_len=5, mixed_len=False, reverse_result=True, scratchpad_type="none")
    for i in range(100):
        prompt, response = s.get_prompt_and_response()
        print(prompt, response)
        validate_addition(prompt, response, reverse_result=True)

def test_sampler_add_mixed():
    s = samplers.SamplerAdd(max_len=5, mixed_len=True, reverse_result=False, scratchpad_type="none")
    for i in range(100):
        prompt, response = s.get_prompt_and_response()
        print(prompt, response)
        validate_addition(prompt, response)

def test_sampler_add_scratchpad():
    s = samplers.SamplerAdd(max_len=5, mixed_len=True, reverse_result=False, scratchpad_type="basic")
    for i in range(100):
        prompt, response = s.get_prompt_and_response()
        print(prompt, response)
        validate_addition(prompt, response)

def test_sampler_mul():
    s = samplers.SamplerMul(max_len=3, mixed_len=False, reverse_result=True, scratchpad_type="none")
    print(s)
        
    for i in range(5):
        print(s.get_prompt_and_response())

def test_sampler_mul_mixed():
    s = samplers.SamplerMul(max_len=3, mixed_len=True, reverse_result=False, scratchpad_type="none")
    print(s)
        
    for i in range(5):
        print(s.get_prompt_and_response())

def test_sampler_mul_scratchpad():
    s = samplers.SamplerMul(max_len=5, mixed_len=True, reverse_result=False, scratchpad_type="basic")
    for i in range(100):
        prompt, response = s.get_prompt_and_response()
        print(prompt, response)


def test_sampler_file():
    s = samplers.SamplerFile(file_name="tests/test_sampler_file.txt")
    possible_samples = [("1+1=", "2"), ("1+2=", "3"), ("12+34=", "46")]
    print(s)
    assert s.get_max_prompt_len() == 6
    assert s.get_max_response_len() == 2
    for i in range(10):
        prompt, response = s.get_prompt_and_response()
        print(prompt, response)
        assert (prompt, response) in possible_samples
    
