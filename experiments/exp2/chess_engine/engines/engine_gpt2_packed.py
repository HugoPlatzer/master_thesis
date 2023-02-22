import engine
import chess
import logging
import torch
import transformers
import os


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
    model_input = [start_token]
    model_input += [pack_move(m) for m in s.split()]
    model_input = torch.tensor([model_input])
    if torch.cuda.is_available():
        model_input = model_input.to("cuda")
    return model_input

def convert_model_output(model_input, model_output):
    a = model_output[0][len(model_input[0]):].item()
    m = unpack_move(a)
    try:
        return unpack_move(a)
    except Exception as e:
        return "fail_invalid"



def move_func(board):
    logging.debug(f"generating for: '{board.fen()}'")
    legal_moves = set(m.uci() for m in board.legal_moves)
    model_input_s = " ".join(m.uci() for m in board.move_stack)
    model_input = prepare_model_input(model_input_s)
    if model_input.shape[1] > max_input_len:
        logging.debug("fail: input too long")
        return "FAIL_LONG_INPUT"
    
    for i in range(num_attempts):
        model_output = model.generate(model_input, min_new_tokens=1, max_new_tokens=1, do_sample=True)
        model_output_s = convert_model_output(model_input, model_output)
        logging.debug(f"attempt {i+1}/{num_attempts}: {model_output_s}")
        if model_output_s in legal_moves:
            logging.debug(f"{model_output_s} legal :)")
            return model_output_s
        else:
            logging.debug(f"{model_output_s} illegal :(")
    logging.debug("fail: no valid attempts")
    return model_output_s


if __name__ == "__main__":
    model_path = os.getenv("GPT2_MODEL_PATH_PACKED")
    if model_path is None:
        raise Exception("please specify model path")
    num_attempts = 1
    model = transformers.GPT2LMHeadModel.from_pretrained(model_path)
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    max_input_len = 200
    
    engine_name = "MyEngine-gpt2-packed"
    engine.enable_logging(engine_name)
    e = engine.Engine(engine_name, move_func)
    e.run()
