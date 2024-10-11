import json

import sampler
import model
import trainer

class Experiment:
    def __init__(self, json_file):
        self.load_from_json(json_file)
    
    def __str__(self):
        return (f"{self.__class__.__name__}("
        f"sampler={self.sampler}, "
        f"model={self.model}, "
        f"trainer={self.trainer})"
        )
    
    def load_from_json(self, json_file):
        json_data = json.loads(open(json_file).read())
        
        sampler_class_name = json_data["sampler"]["type"]
        sampler_params = json_data["sampler"]["params"]
        sampler_class = getattr(sampler, sampler_class_name)
        self.sampler = sampler_class(**sampler_params)
        
        model_class_name = json_data["model"]["type"]
        model_params = json_data["model"]["params"]
        model_class = getattr(model, model_class_name)
        max_prompt_len = self.sampler.get_max_prompt_len()
        max_response_len = self.sampler.get_max_response_len()
        self.model = model_class(
            max_prompt_len=max_prompt_len,
            max_response_len=max_response_len,
            **model_params
        )
        
        trainer_class_name = json_data["trainer"]["type"]
        trainer_params = json_data["trainer"]["params"]
        trainer_class = getattr(trainer, trainer_class_name)
        self.trainer = trainer_class(
            sampler=self.sampler,
            model=self.model,
            **trainer_params
        )
    
    def run_training(self):
        self.trainer.run_training()