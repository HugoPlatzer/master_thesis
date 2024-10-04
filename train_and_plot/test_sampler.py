import sampler

s = sampler.SamplerStringReverse(max_len=5, mixed_len=True)

for i in range(5):
    print(s.get_prompt_and_responseimport model

m = model.GPT2Model(max_prompt_len=8, max_response_len=8, n_embd=768, n_layer=12, n_head=12)
print(m.encode_prompt("a:"))
print(m.encode_prompt("abc:"))
print(m.encode_training_sample("a:", "a"))
print(m.encode_training_sample("abc:", "cba"))())