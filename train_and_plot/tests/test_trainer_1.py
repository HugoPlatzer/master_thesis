import samplers
import model
import trainer

def test():
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
    
    print(t)
    t.build_training_dataset()
    print(t.training_dataset.size())
    print(t.training_dataset[0])
    print(t.training_dataset[-1])
