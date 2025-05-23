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
        
        val_test_sampler_class = getattr(samplers,
                self.config["sampler"]["name"])
        val_test_sampler_params = self.config["sampler"]["params"]
        val_test_sampler = val_test_sampler_class(
                **val_test_sampler_params)
        self.val_sampler = val_test_sampler
        self.test_sampler = val_test_sampler

        if "sampler_train" in self.config:
            train_sampler_class = getattr(samplers,
                    self.config["sampler_train"]["name"])
            train_sampler_params = self.config["sampler_train"]["params"]
            self.train_sampler = train_sampler_class(
                    **train_sampler_params)
        else:
            self.train_sampler = val_test_sampler            
        
        n_positions = self.config["model_params"]["n_positions"]

        self.train_dataset = create_dataset(
            self.train_sampler,
            self.tokenizer,
            self.config["dataset_params"]["train_dataset_size"],
            n_positions
        )
        self.val_dataset = create_dataset(
            self.val_sampler,
            self.tokenizer,
            self.config["dataset_params"]["val_dataset_size"],
            n_positions
        )
        self.test_dataset = create_dataset(
            self.test_sampler,
            self.tokenizer,
            self.config["dataset_params"]["test_dataset_size"],
            n_positions
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
