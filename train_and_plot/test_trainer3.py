import sampler, model, trainer

sampler = sampler.SamplerStringReverse(max_len=5, mixed_len=True)
model = model.GPT2Model(
    max_prompt_len=sampler.get_max_prompt_len(),
    max_response_len=sampler.get_max_response_len(),
    n_embd=128,
    n_layer=2,
    n_head=2,
)
t = trainer.Trainer(
    sampler,
    model,
    training_steps=1000,
    batch_size=2,
    eval_rate=1,
)


sample = model.encode_training_sample("abc:", "cba")
print("loss before training:", model.forward(sample).loss.item())
t.run_training()
print("loss after training:", model.forward(sample).loss.item())