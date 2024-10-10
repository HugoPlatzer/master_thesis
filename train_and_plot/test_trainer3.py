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
    training_steps=10000,
    batch_size=8,
    eval_rate=100,
    accuracy_samples=100
)

t.run_training()
model.save_to_file("model_stringreverse.bin")