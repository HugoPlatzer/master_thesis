import json

import samplers
import model
import evaluator
import trainer

class Experiment:
    def __init__(self, json_file):
        self.load_from_json(json_file)
    
    def get_params(self):
        return {}
    
    def __str__(self):
        params = {
            "sampler": self.sampler,
            "model": self.model,
            "evaluator": self.evaluator,
            "trainer": self.trainer
        }
        params.update(self.get_params())
        params_str = ", ".join(
            f"{name}={value}" for name, value in params.items())
        return f"{self.__class__.__name__}({params_str}"")"
    
    def load_from_json(self, json_file):
        json_data = json.loads(open(json_file).read())
        
        sampler_class_name = json_data["sampler"]["type"]
        sampler_params = json_data["sampler"]["params"]
        sampler_class = getattr(samplers, sampler_class_name)
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
        
        evaluator_class_name = json_data["evaluator"]["type"]
        evaluator_params = json_data["evaluator"]["params"]
        evaluator_class = getattr(evaluator, evaluator_class_name)
        self.evaluator = evaluator_class(self.sampler, self.model,
            **evaluator_params)
        
        trainer_class_name = json_data["trainer"]["type"]
        trainer_params = json_data["trainer"]["params"]
        trainer_class = getattr(trainer, trainer_class_name)
        self.trainer = trainer_class(
            sampler=self.sampler,
            model=self.model,
            evaluator=self.evaluator,
            **trainer_params
        )
    
    def save_results(self, json_file):
        if self.trainer.training_states is not None:
            training_states_data = [ts.get_params()
                for ts in self.trainer.training_states]
        else:
            training_states_data = None
        json_data = {
            "sampler": {
                "type": self.sampler.__class__.__name__,
                "params": self.sampler.get_params()
            },
            "model": {
                "type": self.model.__class__.__name__,
                "params": self.model.get_params()
            },
            "evaluator": {
                "type": self.evaluator.__class__.__name__,
                "params": self.evaluator.get_params()
            },
            "trainer": {
                "type": self.trainer.__class__.__name__,
                "params": self.trainer.get_params()
            },
            "training_states": training_states_data
        }
        with open(json_file, "w") as f:
            json.dump(json_data, f, indent=4)
            f.write("\n")
    
    def run_training(self):
        self.trainer.run_training()
