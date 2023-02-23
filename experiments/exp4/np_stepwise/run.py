import sys
import os
import pickle
import numpy as np
from time import time


def get_initial_state(pos, token):
    return params["wte"][token] + params["wpe"][pos]


def normalize(x, norm_w, norm_b):
    x_norm = (x - np.mean(x)) / np.sqrt(np.var(x) + 1e-5)
    return x_norm * norm_w + norm_b


def do_softmax(x):
    x = np.exp(x - np.max(x))
    return x / sum(x)


def do_attention(state, pos, block):
    qkv = np.matmul(state, params["block"][block]["c_attn_w"]) + params["block"][block]["c_attn_b"]
    query = qkv[0*hidden_size:1*hidden_size]
    key = qkv[1*hidden_size:2*hidden_size]
    value = qkv[2*hidden_size:3*hidden_size]
    query = query.reshape((n_heads, head_size))
    key = key.reshape((n_heads, head_size))
    value = value.reshape((n_heads, head_size))
    prev_states[pos][block]["k"] = key
    prev_states[pos][block]["v"] = value
    attn_outputs = []
    for head_nr in range(n_heads):
        attn_weights = np.array([np.matmul(query[head_nr], prev_states[pos_i][block]["k"][head_nr]) for pos_i in range(pos + 1)])
        attn_weights *= scaling_factor
        attn_weights = do_softmax(attn_weights)
        # ~ print(pos, block, head_nr, attn_weights)
        attn_output = np.sum([attn_weights[pos_i] * prev_states[pos_i][block]["v"][head_nr] for pos_i in range(pos + 1)], axis=0)
        attn_outputs.append(attn_output)
    attn_outputs = np.concatenate(attn_outputs)
    attn_proj = np.matmul(attn_outputs, params["block"][block]["c_proj_w"]) + params["block"][block]["c_proj_b"]
    return attn_proj


def gelu_activation(state):
    return  0.5 * state * (1 + np.tanh(np.sqrt(2 / np.pi) * (state + 0.044715 * (state**3))))


def do_mlp(state, block):
    state = np.matmul(state, params["block"][block]["mlp_fc_w"]) + params["block"][block]["mlp_fc_b"]
    state = gelu_activation(state)
    state = np.matmul(state, params["block"][block]["mlp_proj_w"]) + params["block"][block]["mlp_proj_b"]
    return state
    

def do_gpt2_block(block_inputs, pos, block):
    x = block_inputs
    block_params = params["block"][block]
    x_ln1 = normalize(x, block_params["ln1_w"], block_params["ln1_b"])
    x_ln1_attn = do_attention(x_ln1, pos, block)
    x += x_ln1_attn
    x_ln2 = normalize(x, block_params["ln2_w"], block_params["ln2_b"])
    x_ln2_mlp = do_mlp(x_ln2, block)
    x += x_ln2_mlp
    return x


def do_gpt2_pos(pos, token):
    if len(prev_states) < pos + 1:
        prev_states.append([{"k": None, "v": None} for block in range(n_blocks)])
    state = get_initial_state(pos, token)
    for block in range(n_heads):
        state = do_gpt2_block(state, pos, block)
    state = normalize(state, params["lnf_w"], params["lnf_b"])
    logits = np.matmul(state, params["lm_head_w"].transpose())
        
    # ~ if pos == len(input_str) - 1:
        # ~ pickle.dump(logits, open("state.pickle", "wb"))
    return logits


def print_prediction_values(logits):
    a = [(x, i) for i, x in enumerate(logits)]
    a.sort()
    for value, i in a:
        if chr(i).isprintable():
            print(value, i, f"'{chr(i)}'")
        else:
            print(value, i)



params_file = os.getenv("HF_PARAMS_FILE")
assert params_file is not None
params = pickle.load(open(params_file, "rb"))

hidden_size = 768
n_blocks = 12
n_heads = 12
head_size = hidden_size // n_heads
scaling_factor = 1 / np.sqrt(head_size)



input_str = sys.argv[1]
# prev_states[pos][blocknr]["k" or "v"]
prev_states = []

t1 = time()
for pos in range(len(input_str)):
    logits = do_gpt2_pos(pos, ord(input_str[pos]))
t2 = time()
print("t={:.5f}s".format(t2-t1))

print_prediction_values(logits)
