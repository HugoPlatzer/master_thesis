from transformers import GPT2LMHeadModel
import torch
import sys


def pack_move(m):
    m = m.lower()
    x1 = ord(m[0]) - ord("a")
    x2 = ord(m[1]) - ord("1")
    x3 = ord(m[2]) - ord("a")
    x4 = ord(m[3]) - ord("1")
    if len(m) == 5:
        mapping = {"q": 1, "r": 2, "b": 3, "n": 4}
        x5 = mapping[m[4]]
    else:
        x5 = 0
    return x1 + x2 * 8 + x3 * (8**2) + x4 * (8**3) + x5 * (8**4)


def unpack_move(x):
    x, x1 = x // 8, x % 8
    x, x2 = x // 8, x % 8
    x, x3 = x // 8, x % 8
    x, x4 = x // 8, x % 8
    x, x5 = x // 8, x % 8
    s1 = "abcdefgh"[x1]
    s2 = "12345678"[x2]
    s3 = "abcdefgh"[x3]
    s4 = "12345678"[x4]
    mapping = {0: "", 1: "q", 2: "r", 3: "b", 4: "n"}
    s5 = mapping[x5]
    return s1 + s2 + s3 + s4 + s5

def prepare_model_input(s):
    start_token = 32767
    return torch.tensor([[start_token] + [pack_move(m) for m in s.split()]])

def convert_model_output(model_input, model_output):
    a = model_output[0][len(model_input[0]):].item()
    m = unpack_move(a)
    try:
        return unpack_move(a)
    except Exception as e:
        print(f"invalid move {a}")

model_path = sys.argv[1]
model_input_s = sys.argv[2]


model = GPT2LMHeadModel.from_pretrained(model_path)
model_input = prepare_model_input(model_input_s)
# ~ print(model_input)
model_output = model.generate(model_input, min_new_tokens=1, max_new_tokens=1, do_sample=True)
# ~ print(model_output)
model_output_s = convert_model_output(model_input, model_output)
print(repr(model_output_s))
