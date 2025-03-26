import math
from samplers import SamplerStringReverse, SamplerAdd, SamplerMul, SamplerSqrt, get_sample_int

def test_sampler_stringreverse():
    sampler = SamplerStringReverse(string_length=3)
    for i in range(5):
        sample = sampler.get_sample()
        assert sample["prompt"][-1] == ":"
        assert sample["prompt"][:-1][::-1] == sample["response"]

def test_sampler_add():
    samplers = [SamplerAdd(digits=5),
            SamplerAdd(digits=5, sampling_strategy="uniform_digits")]
    for sampler in samplers:
        for i in range(5):
            sample = sampler.get_sample()
            print(sample)

def test_sampler_mul():
    samplers = [SamplerMul(digits=3),
            SamplerMul(digits=3, sampling_strategy="uniform_digits")]
    for sampler in samplers:
        for i in range(5):
            sample = sampler.get_sample()
            print(sample)

def test_sampler_sqrt():
    samplers = [SamplerSqrt(digits=5), SamplerSqrt(digits=6, sampling_strategy="uniform_digits")]
    for sampler in samplers:
        for i in range(5):
            sample = sampler.get_sample()
            print(sample)
            a = int(sample["prompt"][:-1])
            k = int(sample["response"])
            assert k**2 <= a
            assert (k+1)**2 > a

def test_get_sample_int():
    num_digits = 5
    for sampling_strategy in ["basic", "from_zero", "uniform_digits", "uniform_bits"]:
        for i in range(10):
            x = get_sample_int(num_digits, sampling_strategy)
            print(sampling_strategy, x)
