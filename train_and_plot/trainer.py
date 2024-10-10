import evaluator

import torch
import transformers
import tqdm

class Trainer:
    def __init__(self, sampler, model, training_steps,
        batch_size, eval_rate, accuracy_samples):
        self.sampler = sampler
        self.model = model
        self.training_steps = training_steps
        self.batch_size = batch_size
        self.eval_rate = eval_rate
        self.accuracy_samples = accuracy_samples
    
    def __str__(self):
        return (f"{self.__class__.__name__}("
        f"sampler={self.sampler}, "
        f"model={self.model}], "
        f"training_steps={self.training_steps}, "
        f"batch_size={self.batch_size}, "
        f"eval_rate={self.eval_rate}, "
        f"accuracy_samples={self.accuracy_samples})"
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
    
    # initialize training data batches, optimizer, learning rate scheduler,
    # model evaluator
    def prepare_for_training(self):
        self.build_training_dataset()
        # AdamW with default learning rate as in 'transformers' Trainer class
        self.optimizer = torch.optim.AdamW(
            self.model.model.parameters(), lr=1e-5)
        # learning rate scheduler
        self.lr_scheduler = transformers.get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.training_steps
            )
        self.evaluator = evaluator.Evaluator(self.sampler, self.model,
            num_samples=self.accuracy_samples)
    
    def run_training(self):
        self.prepare_for_training()
        
        for step in tqdm.tqdm(range(1, self.training_steps + 1)):
            inputs = self.training_dataset[step-1]
            outputs = self.model.forward(inputs)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            if step % self.eval_rate == 0:
                loss_value = loss.item()
                accuracy_value = self.evaluator.evaluate_model()
                print(f"step={step} loss={loss_value} "
                      f"accuracy={accuracy_value}")