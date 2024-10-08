import torch

class Trainer:
    def __init__(self, sampler, model, training_steps, batch_size, eval_rate):
        self.sampler = sampler
        self.model = model
        self.training_steps = training_steps
        self.batch_size = batch_size
        self.eval_rate = eval_rate
    
    def __str__(self):
        return (f"{self.__class__.__name__}("
        f"sampler={self.sampler}, "
        f"model={self.model}], "
        f"training_steps={self.training_steps}, "
        f"batch_size={self.batch_size}, "
        f"eval_rate={self.eval_rate})"
        )
    
    def get_sample_len(self):
        return self.model.config.n_positions
    
    def build_training_dataset(self):
        dataset_size = (self.training_steps,
            self.batch_size,
            self.get_sample_len())
        self.training_dataset = torch.zeros(dataset_size, dtype=torch.int64)
        for i in range(self.training_steps):
            for j in range(self.batch_size):
                prompt, response = self.sampler.get_prompt_and_response()
                sample = self.model.encode_training_sample(prompt, response)
                self.training_dataset[i][j] = sample