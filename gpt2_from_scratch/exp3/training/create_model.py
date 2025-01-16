from transformers import GPT2Config, GPT2LMHeadModel


config = GPT2Config(vocab_size=300, max_length=800, eos_token_id=256, pad_token_id=257)
model = GPT2LMHeadModel(config)

print(model)
print(model.num_parameters(), "params")
model.save_pretrained("model/")

