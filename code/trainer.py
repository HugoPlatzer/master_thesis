import sys
import os
from transformers import (Trainer, TrainingArguments, TrainerCallback,
    EarlyStoppingCallback)
from transformers.utils import logging as transformers_logging

from model import load_model_from_path, evaluate_loss, evaluate_accuracy
    
class PerformanceTrackingCallback(TrainerCallback):
    def __init__(self, experiment_results, tokenizer,
            train_dataset, val_dataset, test_dataset,
            logging_compute_accuracy,
            strip_intermediate):
        super().__init__()
        self.experiment_results = experiment_results
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.logging_compute_accuracy = logging_compute_accuracy
        self.strip_intermediate = strip_intermediate
        
        self.best_model_path = None
        self.best_model_step = None
        self.best_model_epoch = None

    def on_log(self, args, state, control, **kwargs):
        step = state.global_step
        epoch = state.epoch
        model = kwargs["model"]
        batch_size = args.per_device_eval_batch_size
        trainer_logs = kwargs["logs"]
        if "loss" in trainer_logs:
            avg_batch_loss = trainer_logs["loss"]
            self.experiment_results.log_metrics(step, {
                    "epoch": epoch,
                    "avg_batch_loss": avg_batch_loss
            })
            if self.logging_compute_accuracy:
                val_accuracy = evaluate_accuracy(
                    model, self.tokenizer, self.val_dataset,
                    batch_size,
                    self.strip_intermediate)
                self.experiment_results.log_metrics(step, {
                    "val_accuracy": val_accuracy
                })
    
    def on_evaluate(self, args, state, control, **kwargs):
        step = state.global_step
        epoch = state.epoch
        model = kwargs["model"]
        batch_size = args.per_device_eval_batch_size
        metrics = kwargs["metrics"]
        if "eval_loss" in metrics:
            train_loss = evaluate_loss(model, self.train_dataset, batch_size)
            val_loss = metrics["eval_loss"]
            val_accuracy = evaluate_accuracy(
                model, self.tokenizer, self.val_dataset, batch_size,
                self.strip_intermediate)
            self.experiment_results.log_metrics(step, {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            })
    
    def on_save(self, args, state, control, **kwargs):
        def get_step_from_checkpoint_path(path):
            name = os.path.basename(path)
            step_nr = int(name.split("-")[-1])
            return step_nr
        
        step = state.global_step
        epoch = state.epoch
        best_model_step = get_step_from_checkpoint_path(
            state.best_model_checkpoint)
        if step == best_model_step: # new best model
            self.best_model_path = os.path.relpath(
                state.best_model_checkpoint,
                start=args.output_dir
            )
            self.best_model_step = step
            self.best_model_epoch = epoch
    
    def on_train_end(self, args, state, control, **kwargs):
        best_model = load_model_from_path(state.best_model_checkpoint)
        batch_size = args.per_device_eval_batch_size
        train_loss = evaluate_loss(
            best_model, self.train_dataset, batch_size)
        val_loss = evaluate_loss(
            best_model, self.val_dataset, batch_size)
        test_loss = evaluate_loss(
            best_model, self.test_dataset, batch_size)
        train_accuracy = evaluate_accuracy(
            best_model, self.tokenizer, self.train_dataset, batch_size,
            self.strip_intermediate)
        val_accuracy = evaluate_accuracy(
            best_model, self.tokenizer, self.val_dataset, batch_size,
            self.strip_intermediate)
        test_accuracy = evaluate_accuracy(
            best_model, self.tokenizer, self.test_dataset, batch_size,
            self.strip_intermediate)
        self.experiment_results.log_best_model_metrics({
            "path": self.best_model_path,
            "step": self.best_model_step,
            "epoch": self.best_model_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy
        })

def create_trainer(model, tokenizer, train_dataset, val_dataset, test_dataset,
        experiment_results, training_params):

    if "strip_intermediate" not in training_params:
        training_params["strip_intermediate"] = False

    training_args = TrainingArguments(
        output_dir=training_params["output_dir"],
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=training_params["logging_steps"],
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        per_device_train_batch_size=training_params["batch_size"],
        per_device_eval_batch_size=training_params["batch_size"],
        num_train_epochs=training_params["max_epochs"]
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[
            PerformanceTrackingCallback(
                experiment_results,
                tokenizer,
                train_dataset,
                val_dataset,
                test_dataset,
                training_params["logging_compute_accuracy"],
                training_params["strip_intermediate"]
            )
            EarlyStoppingCallback(
                early_stopping_patience= \
                    training_params["early_stopping_patience"],
                early_stopping_threshold= \
                    training_params["early_stopping_threshold"]
            )
        ]
    )
    return trainer
