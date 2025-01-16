from transformers import GPT2Config, GPT2LMHeadModel
import sys

model_type = sys.argv[1]

if model_type == "gpt2_uci":
    config = GPT2Config(vocab_size=128, max_length=1024, eos_token_id=ord(";"), pad_token_id=0)
elif model_type == "gpt2_san":
    config = GPT2Config(vocab_size=128, max_length=1024, eos_token_id=ord(";"), pad_token_id=0)
elif model_type == "gpt2_packed":
    config = GPT2Config(vocab_size=32768, max_length=256, eos_token_id=0, pad_token_id=0)
elif model_type == "gpt2_piecestring":
    config = GPT2Config(vocab_size=128, max_length=128, eos_token_id=ord(";"), pad_token_id=0)
elif model_type == "gpt2_fen":
    config = GPT2Config(vocab_size=128, max_length=128, eos_token_id=ord(";"), pad_token_id=0)
else:
    raise Exception("unknown model type")


model = GPT2LMHeadModel(config)

print(model)
print(model.num_parameters(), "params")
model.save_pretrained("model/")
