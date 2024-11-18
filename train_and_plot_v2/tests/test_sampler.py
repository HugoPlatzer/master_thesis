from samplers import SamplerStringReverse

def test_sampler_stringreverse():
    sampler = SamplerStringReverse(string_length=3)
    for i in range(5):
        sample = sampler.get_sample()
        assert sample["prompt"][-1] == ":"
        assert sample["prompt"][:-1][::-1] == sample["response"]
