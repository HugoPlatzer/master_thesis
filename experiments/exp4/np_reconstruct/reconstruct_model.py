import numpy as np
import pickle
import sys
import os
import timeit


def layer_norm(state, w, b):
    mean = np.mean(state, axis=1)
    mean = np.expand_dims(mean, axis=1)
    var = np.var(state, axis=1)
    var = np.expand_dims(var, axis=1)
    state_norm = (state - mean) / np.sqrt(var + 1e-5)
    state_out = w * state_norm + b
    return state_out


def gpt2_block_attention(state, c_attn_w, c_attn_b, c_proj_w, c_proj_b):
    qkv = np.matmul(state, c_attn_w) + c_attn_b
    qkv = qkv.reshape((n_positions, 3, hidden_size))
    query, key, value = qkv[:, 0, :], qkv[:, 1, :], qkv[:, 2, :]

    query = query.reshape((n_positions, n_heads, head_size))
    query = np.swapaxes(query, 0, 1)
    key = key.reshape((key.shape[0], n_heads, head_size))
    key = np.swapaxes(key, 0, 1)
    value = value.reshape((n_positions, n_heads, head_size))
    value = np.swapaxes(value, 0, 1)

    attn_weights = np.matmul(query, np.swapaxes(key, 1, 2))
    attn_weights *= scaling_factor

    mask_matrix_a = np.tile(np.arange(n_positions), (n_positions, 1))
    mask_matrix_b = np.transpose(mask_matrix_a)
    mask_matrix = mask_matrix_a <= mask_matrix_b
    mask_value = np.finfo(np.float32).min

    attn_weights = np.where(mask_matrix, attn_weights, mask_value)
    attn_weights = np.exp(attn_weights) / np.sum(np.exp(attn_weights), axis=2, keepdims=True)

    attn = np.matmul(attn_weights, value)
    attn_merged = np.swapaxes(attn, 0, 1).reshape((n_positions, hidden_size))
    attn_proj = np.matmul(attn_merged, c_proj_w) + c_proj_b
    
    return attn_proj


def gelu_activation(state):
    return  0.5 * state * (1 + np.tanh(np.sqrt(2 / np.pi) * (state + 0.044715 * (state**3))))


def gpt2_block_mlp(state, mlp_fc_w, mlp_fc_b, mlp_proj_w, mlp_proj_b):
    state = np.matmul(state, mlp_fc_w) + mlp_fc_b
    state = gelu_activation(state)
    state = np.matmul(state, mlp_proj_w) + mlp_proj_b
    return state


def gpt2_block(state, block_params):
    state_ln1 = layer_norm(state, block_params["ln1_w"], block_params["ln1_b"])
    state_ln1_attn = gpt2_block_attention(state_ln1, block_params["c_attn_w"], block_params["c_attn_b"], block_params["c_proj_w"], block_params["c_proj_b"])
    state = state + state_ln1_attn

    state_ln2 = layer_norm(state, block_params["ln2_w"], block_params["ln2_b"])
    state_ln2_mlp = gpt2_block_mlp(state_ln2, block_params["mlp_fc_w"], block_params["mlp_fc_b"], block_params["mlp_proj_w"], block_params["mlp_proj_b"])
    
    state = state + state_ln2_mlp
    return state


def gpt2_model(input_tokenids, params):
    token_vectors = params["wte"].take(input_tokenids, axis=0)
    pos_range = np.arange(model_input.shape[0])
    pos_vectors = params["wpe"].take(pos_range, axis=0)
    state = token_vectors + pos_vectors
    
    for i in range(n_blocks):
        state = gpt2_block(state, params["block"][i])

    state = layer_norm(state, params["lnf_w"], params["lnf_b"])
    state = np.matmul(state, params["lm_head_w"].transpose())
    return state


def print_prediction_values(model_output):
    a = [(x, i) for i, x in enumerate(model_output[-1])]
    a.sort()
    for value, i in a:
        if chr(i).isprintable():
            print(value, i, f"'{chr(i)}'")
        else:
            print(value, i)



model_input_s = sys.argv[1]
model_input = np.array([ord(c) for c in model_input_s])

params_file = os.getenv("HF_PARAMS_FILE")
if params_file is None:
    raise Exception("no params file given")
params = pickle.load(open(params_file, "rb"))

n_blocks = 12
n_heads = 12
head_size = 64
hidden_size = n_heads * head_size
scaling_factor = 1 / np.sqrt(head_size)
n_positions = len(model_input)

time_taken = timeit.timeit(lambda: gpt2_model(model_input, params), number=1)
print(time_taken / 1)

model_output = gpt2_model(model_input, params)
# ~ pickle.dump(model_output, open("state.pickle", "wb"))
print_prediction_values(model_output)
