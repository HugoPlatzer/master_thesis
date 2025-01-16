import transformers

config = transformers.GPT2Config(vocab_size=128, max_length=1024, eos_token_id=ord(";"), pad_token_id=0)
m = transformers.GPT2LMHeadModel(config)

print(m.transformer.wpe.weights)
