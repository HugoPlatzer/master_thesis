from transformers import GPT2LMHeadModel, GPT2Config
import torch

import util

class GPT2Model:
    def __init__(self, max_prompt_len, max_response_len, num_sequences,
        n_embd, n_layer, n_head):
        self.max_prompt_len = max_prompt_len
        self.max_response_len = max_response_len
        self.pad_token_id = 0
        # support training with multiple sequential examples in one
        # training sample, even though prompting only uses a single example
        self.num_sequences = num_sequences
        # model should be able to handle max prompt plus response len plus eos token
        n_positions = max_prompt_len + max_response_len + 1
        n_positions *= num_sequences
        self.config = GPT2Config(
            vocab_size=256,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
        )
        self.model = GPT2LMHeadModel(self.config)
        self.model = util.move_tensor_to_device(self.model)
    
    @classmethod
    def load_from_file(cls, filename):
        return torch.load(filename)
    
    def save_to_file(self, filename):
        torch.save(self, filename)
    
    def get_params(self):
        return {
            "max_prompt_len" : self.max_prompt_len,
            "max_response_len" : self.max_response_len,
            "num_sequences" : self.num_sequences,
            "n_positions" : self.config.n_positions,
            "n_embd" : self.config.n_embd,
            "n_layer" : self.config.n_layer,
            "n_head" : self.config.n_head
        }
    
    def __str__(self):
        params_str = ", ".join(
            f"{name}={value}" for name, value in self.get_params().items())
        return f"{self.__class__.__name__}({params_str}"")"

    def encode_and_pad(self, s, max_len, pad_direction):
        if len(s) > max_len:
            raise ValueError("string too long")
        x = [ord(c) for c in s]
        n_pad = max_len - len(x)
        if pad_direction == "left":
            x = ([self.pad_token_id] * n_pad) + x
        elif pad_direction == "right":
            x = x + ([self.pad_token_id] * n_pad)
        else:
            raise ValueError("invalid pad direction")
        x = torch.tensor(x, dtype=torch.int64)
        return x
    
    def encode_prompt(self, prompt_str):
        return self.encode_and_pad(prompt_str, self.max_prompt_len, "left")
    
    def encode_training_sample(self, prompts_responses):
        sample_seqs = []
        if len(prompts_responses) != self.num_sequences:
            raise ValueError(f"need {self.num_sequences} prompts and responses")
        for prompt_str, response_str in prompts_responses:
            prompt = self.encode_and_pad(
                prompt_str, self.max_prompt_len, "left")
            response = self.encode_and_pad(
                response_str, self.max_response_len, "right")
            end_token = torch.tensor([self.pad_token_id], dtype=torch.int64)
            sample_seq = torch.cat((prompt, response, end_token))
            sample_seqs.append(sample_seq)
        sample = torch.cat(sample_seqs)
        assert sample.shape == (self.config.n_positions,)
        return sample
    
    def decode_response(self, response):
        response = response.tolist()
        response = [c for c in response if c != self.pad_token_id]
        response_str = "".join(chr(c) for c in response)
        return response_str
    
    def answer_prompts(self, prompt_strings):
        # ensure model is in eval mode, which disables dropout and ensures
        # deterministic generation
        self.model.eval()
        prompts = torch.stack([self.encode_prompt(s) for s in prompt_strings])
        prompts = util.move_tensor_to_device(prompts)
        # create dummy attention mask (attention on all positions is enabled)
        # to fix an issue on some devices
        dummy_attention_mask = torch.ones_like(prompts)
        # generate response tokens using greedy decoding
        # up to max response length of model (plus eos token)
        responses = self.model.generate(prompts,
            max_new_tokens=self.max_response_len + 1,
            do_sample=False,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.pad_token_id,
            attention_mask=dummy_attention_mask)
        # strip prompt part to leave only response
        responses = responses[:, self.max_prompt_len:]
        response_strings = [self.decode_response(x) for x in responses]
        return response_strings
    
    # forward sequences through model for training purposes
    def forward(self, inputs):
        self.model.train()
        # use labels as inputs
        # left shifting of labels relative to inputs happens inside
        # transformers GPT2 model
        return self.model(inputs, labels=inputs)
