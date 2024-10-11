import sampler
import model
import trainer

def run_test():
    sampler_ = sampler.SamplerStringReverse(max_len=5, mixed_len=True)
    model_ = model.GPT2Model(
        max_prompt_len=sampler_.get_max_prompt_len(),
        max_response_len=sampler_.get_max_response_len(),
        n_embd=128,
        n_layer=2,
        n_head=2,
    )
    t = trainer.Trainer(
        sampler_,
        model_,
        training_steps=10,
        batch_size=2,
        eval_rate=1,
        accuracy_samples=100
    )
    
    print(t)
    t.build_training_dataset()
    print(t.training_dataset.size())
    print(t.training_dataset[0])
    print(t.training_dataset[-1])