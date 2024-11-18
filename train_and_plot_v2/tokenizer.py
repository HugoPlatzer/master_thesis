from transformers import PreTrainedTokenizer

class ASCIITokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        self.vocab = {chr(i): i for i in range(128)}
        self.vocab_size = 128
        super().__init__(**kwargs)
        self.my_eos_token_id = 0
        self.my_pad_token_id = 0
    
    def get_vocab(self):
        return self.vocab
    
    def vocab_size(self):
        return len(self.vocab)
    
    def encode(self, text, add_special_tokens=True):
        if add_special_tokens:
            return [ord(c) for c in text] + [self.my_eos_token_id]
        else:
            return [ord(c) for c in text]
    
    def decode(self, token_ids, skip_special_tokens=True):
        if skip_special_tokens:
            return "".join(chr(x) for x in token_ids
                if x not in (self.my_eos_token_id, self.my_pad_token_id))
        else:
            return "".join(chr(x) for x in token_ids)
