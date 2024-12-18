import samplers
import model
import evaluator
import trainer

def test():
    sampler = samplers.SamplerStringReverse(max_len=1, mixed_len=False)
    model_ = model.GPT2Model(
        max_prompt_len=sampler.get_max_prompt_len(),
        max_response_len=sampler.get_max_response_len(),
        n_embd=128,
        n_layer=2,
        n_head=2,
    )
    evaluator_ = evaluator.Evaluator(sampler, model_, num_samples=100, strip_scratchpad=False)
    t = trainer.Trainer(
        sampler,
        model_,
        evaluator_,
        training_steps=100,
        batch_size=2,
        eval_rate=10,
        lr_initial_value=1e-5,
        lr_warmup_steps=0,
        lr_scheduler_type="linear",
    )
    
    t.run_training()
    for ts in t.training_states:
        print(ts)
