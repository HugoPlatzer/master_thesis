import torch
import transformers
import tqdm

import util
import evaluator
import training_state

class Trainer:
    def __init__(self, sampler, model, evaluator, training_steps, batch_size,
        eval_rate, lr_initial_value, lr_warmup_steps, lr_scheduler_type):
        self.sampler = sampler
        self.model = model
        self.evaluator = evaluator
        self.training_steps = training_steps
        self.batch_size = batch_size
        self.eval_rate = eval_rate
        self.lr_initial_value = lr_initial_value
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_scheduler_type = lr_scheduler_type
        
        self.training_dataset = None
        self.optimizer = None
        self.lr_scheduler = None
        self.training_states = None
    
    def get_params(self):
        return {
            "training_steps": self.training_steps,
            "batch_size": self.batch_size,
            "eval_rate": self.eval_rate,
            "lr_initial_value": self.lr_initial_value,
            "lr_warmup_steps": self.lr_warmup_steps,
            "lr_scheduler_type": self.lr_scheduler_type
        }
    
    def __str__(self):
        params = {
            "sampler": self.sampler,
            "model": self.model,
            "evaluator": self.evaluator
        }
        params.update(self.get_params())
        params_str = ", ".join(
            f"{name}={value}" for name, value in params.items())
        return f"{self.__class__.__name__}({params_str}"")"
    
    def get_sample_len(self):
        return self.model.config.n_positions
    
    def build_training_dataset(self):
        print("building training dataset...")
        dataset_size = (self.training_steps,
            self.batch_size,
            self.get_sample_len())
        self.training_dataset = torch.zeros(dataset_size, dtype=torch.int64)
        for i in tqdm.tqdm(range(self.training_steps)):
            for j in range(self.batch_size):
                prompt, response = self.sampler.get_prompt_and_response()
                sample = self.model.encode_training_sample(prompt, response)
                self.training_dataset[i][j] = sample
    
    # initialize training dataset, optimizer, learning rate scheduler,
    # training states
    def prepare_for_training(self):
        self.build_training_dataset()
        self.optimizer = torch.optim.AdamW(
            self.model.model.parameters(), lr=self.lr_initial_value)
        self.lr_scheduler = transformers.get_scheduler(
            name=self.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=self.training_steps
            )
        self.training_states = []
    
    def run_training(self):
        self.prepare_for_training()
        
        print("running training...")
        for step in tqdm.tqdm(range(1, self.training_steps + 1)):
            inputs = self.training_dataset[step-1]
            outputs = self.model.forward(inputs)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            if step == 1 or step % self.eval_rate == 0:
                current_lr = self.lr_scheduler.get_last_lr()[0]
                loss_value = loss.item()
                accuracy_value = self.evaluator.evaluate_model()
                print(f"step={step} lr={current_lr:.8f}"
                    f" loss={loss_value:.6f} accuracy={accuracy_value:.3f}")
                state = training_state.TrainingState(
                    training_step=step,
                    loss=loss_value,
                    accuracy=accuracy_value
                )
                self.training_states.append(state)
        
        print("training complete.")
