from samplers import SamplerStringReverse
from dataset import create_dataset
from tokenizer import ASCIITokenizer

def test_dataset():
    sampler = SamplerStringReverse(string_length=3)
    tokenizer = ASCIITokenizer()
    ds = create_dataset(sampler, tokenizer, num_samples=5)
