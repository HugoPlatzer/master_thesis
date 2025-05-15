from transformers import (GPT2LMHeadModel, GPT2Config,
    Trainer, TrainingArguments)
import torch
import tqdm


def create_model(tokenizer, n_positions, n_embd, n_layer, n_head):
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        eos_token_id=tokenizer.my_eos_token_id,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head
    )
    model = GPT2LMHeadModel(config)
    return model

def load_model_from_path(path):
    return GPT2LMHeadModel.from_pretrained(path)

def evaluate_loss(model, dataset, batch_size):
    training_args = TrainingArguments(
        output_dir=".",
        per_device_eval_batch_size=batch_size,
        disable_tqdm=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset
    )
    eval_result = trainer.evaluate()
    return eval_result["eval_loss"]

def check_response(model_response, good_response,
        strip_intermediate=False):
    if strip_intermediate:
        model_response = model_response.split("]")[-1]
        good_response = good_response.split("]")[-1]
    return (model_response == good_response)

def evaluate_accuracy_batch(model, tokenizer, prompts, good_responses,
        check_fn):
    prompt_ids = [tokenizer.encode(prompt, add_special_tokens=False)
        for prompt in prompts]
    prompt_ids = torch.tensor(prompt_ids, device=model.device)
    prompt_length = prompt_ids.shape[1]
    model_max_length = model.config.n_positions
    dummy_attention_mask = torch.ones_like(prompt_ids)
    model.eval()
    
    response_ids = model.generate(
        prompt_ids,
        max_length=model_max_length,
        do_sample=False,
        pad_token_id=model.config.eos_token_id,
        attention_mask=dummy_attention_mask
    )
    
    response_ids = response_ids[:, prompt_length:]
    response_strs = [
        tokenizer.decode(response, skip_special_tokens=True)
        for response in response_ids]
    assert len(response_strs) == len(good_responses)
    num_total = len(response_strs)
    assert num_total >= 1
    
    num_good = 0
    for model_response, correct_response in zip(response_strs, good_responses):
        if check_fn(model_response, correct_response):
            num_good += 1
    accuracy = num_good / num_total
    return accuracy

def evaluate_accuracy(model, tokenizer, dataset, batch_size,
        strip_intermediate=False):
    num_batches = 0
    total_accuracy = 0.0
    check_fn = (lambda model_response, correct_response:
            check_response(model_response,
            correct_response, strip_intermediate))

    for batch_start in range(0, len(dataset), batch_size):
        batch_end = batch_start + batch_size
        batch_prompts = dataset[batch_start:batch_end]["prompt_str"]
        batch_responses = dataset[batch_start:batch_end]["response_str"]
        batch_accuracy = evaluate_accuracy_batch(
            model, tokenizer, batch_prompts, batch_responses,
            check_fn)
        total_accuracy += batch_accuracy * len(batch_prompts) / len(dataset)
        num_batches += 1
    return total_accuracy


def generate_responses_batch(model, tokenizer, prompts):
    prompt_ids = [tokenizer.encode(prompt, add_special_tokens=False)
        for prompt in prompts]
    prompt_ids = torch.tensor(prompt_ids, device=model.device)
    prompt_length = prompt_ids.shape[1]
    model_max_length = model.config.n_positions
    dummy_attention_mask = torch.ones_like(prompt_ids, device=model.device)
    model.eval()
    
    response_ids = model.generate(
        prompt_ids,
        max_length=model_max_length,
        do_sample=False,
        pad_token_id=model.config.eos_token_id,
        attention_mask=dummy_attention_mask
    )
    
    response_ids = response_ids[:, prompt_length:]
    response_strs = [
        tokenizer.decode(response, skip_special_tokens=True)
        for response in response_ids]

    return response_strs


def generate_responses(model, tokenizer, prompts, batch_size):
    if torch.cuda.is_available():
        model = model.to("cuda")
    elif torch.backends.mps.is_available():
        model = model.to("mps")

    responses = []
    progress_bar = tqdm.tqdm(total=len(prompts),
            desc="Generating responses")
    for batch_start in range(0, len(prompts), batch_size):
        batch_end = batch_start + batch_size
        batch_prompts = prompts[batch_start:batch_end]
        progress_bar.update(len(batch_prompts))
        batch_responses = generate_responses_batch(
            model, tokenizer, batch_prompts)
        responses.extend(batch_responses)
    progress_bar.close()
    return responses
