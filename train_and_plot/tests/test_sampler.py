import sampler

def run_test():
    s = sampler.SamplerStringReverse(max_len=5, mixed_len=True)
    print(s)
    
    for i in range(5):
        print(s.get_prompt_and_response())