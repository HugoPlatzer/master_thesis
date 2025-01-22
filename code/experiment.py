import json
import os
import random
import numpy as np
import torch

from experiment_results import ExperimentResults
from tokenizer import ASCIITokenizer
import samplers
from dataset import create_dataset
from model import create_model
from trainer import create_trainer


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Experiment:
    def __init__(self, path):
        self.path = path
        
        config_file = os.path.join(path, "config.json")
        self.config = json.loads(open(config_file).read())
        
        log_file = os.path.join(self.path, "logfile.txt")
        results_file = os.path.join(self.path, "results.json")
        self.experiment_results = ExperimentResults(
                self.config, log_file, results_file)
        
        random_seed = self.config["random_seed"]
        set_random_seed(random_seed)

        self.tokenizer = ASCIITokenizer()
        
        sampler_class = getattr(samplers, self.config["sampler"]["name"])
        sampler_params = self.config["sampler"]["params"]
        
        self.sampler = sampler_class(**sampler_params)
        
        self.train_dataset = create_dataset(
            self.sampler,
            self.tokenizer,
            self.config["dataset_params"]["train_dataset_size"]
        )
        self.val_dataset = create_dataset(
            self.sampler,
            self.tokenizer,
            self.config["dataset_params"]["val_dataset_size"]
        )
        self.test_dataset = create_dataset(
            self.sampler,
            self.tokenizer,
            self.config["dataset_params"]["test_dataset_size"]
        )
        
        self.model = create_model(
            self.tokenizer,
            self.config["model_params"]["n_positions"],
            self.config["model_params"]["n_embd"],
            self.config["model_params"]["n_layer"],
            self.config["model_params"]["n_head"]
        )
        
        training_params = dict(self.config["training_params"])
        training_params["output_dir"] = self.path
        self.trainer = create_trainer(
            self.model,
            self.tokenizer,
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            self.experiment_results,
            training_params
        )
        
        
    def run(self):
        self.trainer.train()
        self.experiment_results.save()
