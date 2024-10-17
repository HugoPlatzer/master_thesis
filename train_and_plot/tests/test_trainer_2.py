import samplers
import model
import trainer


sampler = samplers.SamplerStringReverse(max_len=5, mixed_len=True)
model_ = model.GPT2Model(
    max_prompt_len=sampler.get_max_prompt_len(),
    max_response_len=sampler.get_max_response_len(),
    n_embd=128,
    n_layer=2,
    n_head=2,
)
t = trainer.Trainer(
    sampler,
    model_,
    training_steps=10,
    batch_size=2,
    lr_scheduler_type="linear",
    initial_lr=1e-5,
    eval_rate=1,
    accuracy_samples=100
)


t.prepare_for_training()
print(t.optimizer)
t.lr_scheduler.step()
print(t.optimizer)
