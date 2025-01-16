from datasets import Dataset

def create_dataset(sampler, tokenizer, num_samples):
    def data_generator():
        for i in range(num_samples):
            sample = sampler.get_sample()
            input_str = sample["prompt"] + sample["response"]
            input_ids = tokenizer.encode(input_str,
                add_special_tokens=True)
            yield {
                "prompt_str": sample["prompt"],
                "response_str": sample["response"],
                "input_str": input_str,
                "input_ids": input_ids,
                "labels": input_ids
            }
    
    # better use from_list than from_generator
    # due to issues with rng state not being updated acroos multiple datasets
    # and data generation logic updates not reflected due to caching
    return Dataset.from_list(list(data_generator()))
