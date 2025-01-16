from model import create_model
from tokenizer import ASCIITokenizer

def test_model():
    model = create_model(
        tokenizer=ASCIITokenizer(),
        n_positions=16,
        n_embd=384,
        n_layer=6,
        n_head=6)
