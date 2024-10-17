class Sampler:
    def __init__(self, **kwargs):
        self.params = {name: value for name, value in kwargs.items()}
    
    def get_params(self):
        return dict(self.params)
    
    def __str__(self):
        params_str = ", ".join(
            f"{name}={value}" for name, value in self.get_params().items())
        return f"{self.__class__.__name__}({params_str}"")"
    
    def get_prompt_and_response(self):
        pass
    
    def get_max_prompt_len(self):
        pass
    
    def get_max_response_len(self):
        pass
