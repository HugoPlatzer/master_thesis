import json
from datetime import datetime

class ExperimentResults:
    def __init__(self, experiment_config, log_file, results_file):
        self.experiment_config = experiment_config
        self.log_file = log_file
        self.log_file_handle = open(log_file, "w")
        self.results_file = results_file
        self.metrics = {}
        self.best_model_metrics = {}
        self.start_timestamp = datetime.now()
        self.end_timestamp = None
    
    def format_metric(self, metric_name, metric_value):
        if "loss" in metric_name:
            return f"{metric_name}={metric_value:.5f}"
        elif "accuracy" in metric_name:
            return f"{metric_name}={metric_value:.3f}"
        else:
            return f"{metric_name}={metric_value}"
    
    def format_metrics_string(self, metrics):
        return " ".join(self.format_metric(metric_name, metric_value)
            for metric_name, metric_value in metrics.items())
    
    def log_string(self, s):
        print(s)
        print(s, file=self.log_file_handle)
        self.log_file_handle.flush()
    
    def log_metrics(self, step, metrics):
        if step not in self.metrics:
            self.metrics[step] = {}
        self.metrics[step].update(metrics)
        log_string = f"step={step} {self.format_metrics_string(metrics)}"
        self.log_string(log_string)
    
    def log_best_model_metrics(self, metrics):
        self.best_model_metrics = metrics
        log_string = "best_model " + self.format_metrics_string(metrics)
        self.log_string(log_string)
    
    def format_time(self, datetime_obj):
        return datetime_obj.strftime("%Y-%b-%d %H:%M:%S")
    
    def save(self):
        # Assume experiment ends at saving
        self.end_timestamp = datetime.now()
        total_seconds = (self.end_timestamp - self.start_timestamp)\
            .total_seconds()
        
        metrics_list = []
        for step, metrics_at_step in self.metrics.items():
            metrics_dict = {"step" : step}
            metrics_dict.update(metrics_at_step)
            metrics_list.append(metrics_dict)
        json_obj = {
            "config": self.experiment_config,
            "start_time": self.format_time(self.start_timestamp),
            "end_time": self.format_time(self.end_timestamp),
            "total_seconds": f"{total_seconds:.1f}",
            "best_model": self.best_model_metrics,
            "training_metrics": metrics_list
        }
        with open(self.results_file, "w") as f:
            print(json.dumps(json_obj, indent=4), file=f)
