from transformers import GPT2LMHeadModel
import sys
import pickle
import numpy as np
import os


model_path = os.getenv("HF_MODEL_PATH")
if model_path is None:
    raise Exception("model path not given")
out_file = os.getenv("HF_PARAMS_FILE")
if out_file is None:
    raise Exception("params file not given")

model = GPT2LMHeadModel.from_pretrained(model_path)
params = {}
params["wte"] = np.array(model.transformer.wte.weight.detach())
params["wpe"] = np.array(model.transformer.wpe.weight.detach())
params["lnf_w"] = np.array(model.transformer.ln_f.weight.detach())
params["lnf_b"] = np.array(model.transformer.ln_f.bias.detach())
params["lm_head_w"] = np.array(model.lm_head.weight.detach())
params["block"] = []

for block in model.transformer.h:
    block_params = {"ln1_w": np.array(block.ln_1.weight.detach()),
                    "ln1_b": np.array(block.ln_1.bias.detach()),
                    "c_attn_w": np.array(block.attn.c_attn.weight.detach()),
                    "c_attn_b": np.array(block.attn.c_attn.bias.detach()),
                    "c_proj_w": np.array(block.attn.c_proj.weight.detach()),
                    "c_proj_b": np.array(block.attn.c_proj.bias.detach()),
                    "ln2_w": np.array(block.ln_2.weight.detach()),
                    "ln2_b": np.array(block.ln_2.bias.detach()),
                    "mlp_fc_w": np.array(block.mlp.c_fc.weight.detach()),
                    "mlp_fc_b": np.array(block.mlp.c_fc.bias.detach()),
                    "mlp_proj_w": np.array(block.mlp.c_proj.weight.detach()),
                    "mlp_proj_b": np.array(block.mlp.c_proj.bias.detach()),
                    }
    params["block"].append(block_params)

pickle.dump(params, open(out_file, "wb"))
