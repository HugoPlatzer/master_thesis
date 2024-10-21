import samplers

def test_sampler_string_reverse():
    s = samplers.SamplerStringReverse(max_len=5, mixed_len=True)
    print(s)
        
    for i in range(5):
        print(s.get_prompt_and_response())

def test_sampler_add():
    s = samplers.SamplerAdd(max_len=5, mixed_len=False, reverse_result=True)
    print(s)
        
    for i in range(5):
        print(s.get_prompt_and_response())

def test_sampler_add_mixed():
    s = samplers.SamplerAdd(max_len=5, mixed_len=True, reverse_result=False)
    print(s)
        
    for i in range(5):
        print(s.get_prompt_and_response())

def test_sampler_mul():
    s = samplers.SamplerMul(max_len=3, mixed_len=False, reverse_result=True)
    print(s)
        
    for i in range(5):
        print(s.get_prompt_and_response())

def test_sampler_mul_mixed():
    s = samplers.SamplerMul(max_len=3, mixed_len=True, reverse_result=False)
    print(s)
        
    for i in range(5):
        print(s.get_prompt_and_response())