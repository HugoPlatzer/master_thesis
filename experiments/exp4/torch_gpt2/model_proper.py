import torch
import pickle


class GPT2Block_Attention(torch.nn.Module):
    def __init__(self, block_idx, n_hidden, n_heads):
        super().__init__()
        self.block_idx = block_idx
        self.n_hidden = n_hidden
        self.n_heads = n_heads
        self.head_size = n_hidden // n_heads
        self.mods = torch.nn.ModuleDict({"query": torch.nn.Linear(n_hidden, n_hidden),
                                         "key": torch.nn.Linear(n_hidden, n_hidden),
                                         "value": torch.nn.Linear(n_hidden, n_hidden),
                                         "proj": torch.nn.Linear(n_hidden, n_hidden),
                                         "drop1": torch.nn.Dropout(0.1),
                                         "drop2": torch.nn.Dropout(0.1)})
        
    
    
    def load_weights_from_dump(self, dump_file):
        dump = pickle.load(open(dump_file, "rb"))
        dump = dump["block"][self.block_idx]
        self.mods["query"].weight = torch.nn.Parameter(torch.tensor(dump["c_attn_w"][:, 0:self.n_hidden].transpose()))
        self.mods["key"].weight = torch.nn.Parameter(torch.tensor(dump["c_attn_w"][:, self.n_hidden:2*self.n_hidden].transpose()))
        self.mods["value"].weight = torch.nn.Parameter(torch.tensor(dump["c_attn_w"][:, 2*self.n_hidden:3*self.n_hidden].transpose()))
        self.mods["query"].bias = torch.nn.Parameter(torch.tensor(dump["c_attn_b"][0:self.n_hidden]))
        self.mods["key"].bias = torch.nn.Parameter(torch.tensor(dump["c_attn_b"][self.n_hidden:2*self.n_hidden]))
        self.mods["value"].bias = torch.nn.Parameter(torch.tensor(dump["c_attn_b"][2*self.n_hidden:3*self.n_hidden]))
        self.mods["proj"].weight = torch.nn.Parameter(torch.tensor(dump["c_proj_w"].transpose()))
        self.mods["proj"].bias = torch.nn.Parameter(torch.tensor(dump["c_proj_b"]))
    
    
    def attn_part(self, x, name):
        x = self.mods[name](x)
        x = x.view(x.shape[0], x.shape[1], self.n_heads, self.head_size)
        x = x.swapaxes(1, 2)
        return x
    
    
    def forward(self, x):
        query = self.attn_part(x, "query")
        key = self.attn_part(x, "key")
        value = self.attn_part(x, "value")
        
        attn_weights = query.matmul(key.swapaxes(2, 3))
        scaling_factor = 1.0 / (self.head_size ** 0.5)
        attn_weights *= scaling_factor
        
        n_pos = attn_weights.shape[2]
        mask_col = torch.arange(n_pos).tile(n_pos, 1)
        mask_row = mask_col.swapaxes(0, 1)
        mask = mask_col <= mask_row
        
        mask_value = torch.tensor(torch.finfo(attn_weights.dtype).min)
        if torch.cuda.is_available():
            mask = mask.to("cuda")
            mask_value = mask_value.to("cuda")
        
        attn_weights = attn_weights.where(mask, mask_value)
        attn_weights = attn_weights.softmax(dim=3)
        attn_weights = self.mods["drop1"](attn_weights)
        
        attn = attn_weights.matmul(value)
        attn = attn.swapaxes(1, 2)
        attn = attn.reshape(attn.shape[0], n_pos, self.n_hidden)
        
        attn = self.mods["proj"](attn)
        attn = self.mods["drop2"](attn)
        
        return attn



class GPT2Block_MLP(torch.nn.Module):
    def __init__(self, block_idx, n_hidden):
        super().__init__()
        self.block_idx = block_idx
        self.n_hidden = n_hidden
        self.n_hidden_mlp = 4 * n_hidden
        self.mods = torch.nn.ModuleDict({"fc": torch.nn.Linear(self.n_hidden, self.n_hidden_mlp),
                                         "act": torch.nn.GELU(approximate="tanh"),
                                         "proj": torch.nn.Linear(self.n_hidden_mlp, self.n_hidden),
                                         "drop": torch.nn.Dropout(0.1)})
    
    
    def load_weights_from_dump(self, dump_file):
        dump = pickle.load(open(dump_file, "rb"))
        dump = dump["block"][self.block_idx]
        self.mods["fc"].weight = torch.nn.Parameter(torch.tensor(dump["mlp_fc_w"].transpose()))
        self.mods["fc"].bias = torch.nn.Parameter(torch.tensor(dump["mlp_fc_b"]))
        self.mods["proj"].weight = torch.nn.Parameter(torch.tensor(dump["mlp_proj_w"].transpose()))
        self.mods["proj"].bias = torch.nn.Parameter(torch.tensor(dump["mlp_proj_b"]))
    
    
    def forward(self, x):
        x = self.mods["fc"](x)
        x = self.mods["act"](x)
        x = self.mods["proj"](x)
        x = self.mods["drop"](x)
        return x



class GPT2Block(torch.nn.Module):
    def __init__(self, block_idx, n_hidden, n_heads):
        super().__init__()
        self.block_idx = block_idx
        self.n_hidden = n_hidden
        self.n_heads = n_heads
        self.mods = torch.nn.ModuleDict({"ln1": torch.nn.LayerNorm(n_hidden),
                                         "attn": GPT2Block_Attention(block_idx, n_hidden, n_heads),
                                         "ln2": torch.nn.LayerNorm(n_hidden),
                                         "mlp": GPT2Block_MLP(block_idx, n_hidden)})
    
    
    def load_weights_from_dump(self, dump_file):
        dump = pickle.load(open(dump_file, "rb"))
        dump = dump["block"][self.block_idx]
        self.mods["ln1"].weight = torch.nn.Parameter(torch.tensor(dump["ln1_w"]))
        self.mods["ln1"].bias = torch.nn.Parameter(torch.tensor(dump["ln1_b"]))
        self.mods["attn"].load_weights_from_dump(dump_file)
        self.mods["ln2"].weight = torch.nn.Parameter(torch.tensor(dump["ln2_w"]))
        self.mods["ln2"].bias = torch.nn.Parameter(torch.tensor(dump["ln2_b"]))
        self.mods["mlp"].load_weights_from_dump(dump_file)
    
    
    def forward(self, x):
        x_ln1 = self.mods["ln1"](x)
        x_ln1_attn = self.mods["attn"](x_ln1)
        x = x + x_ln1_attn
        x_ln2 = self.mods["ln2"](x)
        x_ln2_mlp = self.mods["mlp"](x_ln2)
        x = x + x_ln2_mlp
        return x



class GPT2(torch.nn.Module):
    def __init__(self, n_pos, n_vocab, n_hidden, n_heads, n_blocks):
        super().__init__()
        self.mods = torch.nn.ModuleDict({"wte": torch.nn.Embedding(n_vocab, n_hidden),
                                         "wpe": torch.nn.Embedding(n_pos, n_hidden),
                                         "lnf": torch.nn.LayerNorm(n_hidden),
                                         "lmhead": torch.nn.Linear(n_hidden, n_vocab, bias=False),
                                         "drop": torch.nn.Dropout(0.1)})
        self.blocks = torch.nn.ModuleList([GPT2Block(i, n_hidden, n_heads) for i in range(n_blocks)])
    
    
    def load_weights_from_dump(self, dump_file):
        dump = pickle.load(open(dump_file, "rb"))
        self.mods["wte"].weight = torch.nn.Parameter(torch.tensor(dump["wte"]))
        self.mods["wpe"].weight = torch.nn.Parameter(torch.tensor(dump["wpe"]))
        self.mods["lnf"].weight = torch.nn.Parameter(torch.tensor(dump["lnf_w"]))
        self.mods["lnf"].bias = torch.nn.Parameter(torch.tensor(dump["lnf_b"]))
        self.mods["lmhead"].weight = torch.nn.Parameter(torch.tensor(dump["lm_head_w"]))
        for block in self.blocks:
            block.load_weights_from_dump(dump_file)
    
    
    def forward(self, x):
        batch_size = x.shape[0]
        batch_npos = x.shape[1]
        
        x_token = self.mods["wte"](x)
        pos_range = torch.arange(batch_npos)
        if torch.cuda.is_available():
            pos_range = pos_range.to("cuda")
        x_pos = self.mods["wpe"](pos_range)
        x = x_token + x_pos
        x = self.mods["drop"](x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.mods["lnf"](x)
        x = self.mods["lmhead"](x)
        
        return x
    
    
    def generate(self, x, num_new_tokens):
        new_tokens = []
        for i in range(num_new_tokens):
            model_in = x.unsqueeze(dim=0)
            model_out = self.forward(model_in)
            new_token_probs = model_out[0][-1]
            new_token = new_token_probs.argmax()
            new_tokens.append(new_token.item())
            x = torch.concatenate((x, new_token.unsqueeze(dim=0)))
        return new_tokens
