from transformers import GPT2Config, GPT2LMHeadModel

# ~ config = GPT2Config()
config = GPT2Config(vocab_size=256, max_length=100, eos_token_id=ord("."), pad_token_id=ord("."))


model = GPT2LMHeadModel(config)

print(model)
print(model.num_parameters(), "params")
model.save_pretrained("model/")
