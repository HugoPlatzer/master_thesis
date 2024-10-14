class TrainingState:
    def __init__(self, training_step, loss, accuracy):
        self.training_step = training_step
        self.loss = loss
        self.accuracy = accuracy
        
    def get_params(self):
        return {
            "training_step": self.training_step,
            "loss": self.loss,
            "accuracy": self.accuracy
        }
    
    def __str__(self):
        params = self.get_params()
        params_str = ", ".join(
            f"{name}={value}" for name, value in params.items())
        return f"{self.__class__.__name__}({params_str}"")"
