from tokenizer import ASCIITokenizer

def test_tokenizer_1():
    t = ASCIITokenizer()
    assert t.encode("abc", add_special_tokens=True) == [97, 98, 99, 0]
    assert t.decode([97, 98, 99], skip_special_tokens=True) == "abc"
