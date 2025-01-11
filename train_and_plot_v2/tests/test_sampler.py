from samplers import SamplerStringReverse, SamplerSqrt

def test_sampler_stringreverse():
    sampler = SamplerStringReverse(string_length=3)
    for i in range(5):
        sample = sampler.get_sample()
        assert sample["prompt"][-1] == ":"
        assert sample["prompt"][:-1][::-1] == sample["response"]

def test_sampler_sqrt():
    samplers = [SamplerSqrt(digits=5), SamplerSqrt(digits=6)]
    for sampler in samplers:
        for i in range(5):
            sample = sampler.get_sample()
            a = int(sample["prompt"][:-1])
            k = int(sample["response"])
            print(a, k)
            assert k**2 <= a
            assert (k+1)**2 > a
