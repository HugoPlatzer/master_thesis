from samplers import SamplerStringReverse
from dataset import create_dataset

def test_dataset():
    sampler = SamplerStringReverse(string_length=3)
    ds = create_dataset(sampler, 5)
