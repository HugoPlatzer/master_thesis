from tokenizer import ASCIITokenizer

def test_tokenizer_1():
    t = ASCIITokenizer()
    assert t.encode("abc") == [97, 98, 99]
    assert t.decode([97, 98, 99]) == "abc"
