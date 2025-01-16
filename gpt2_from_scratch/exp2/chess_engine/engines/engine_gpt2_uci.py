import engine
import chess
import logging
import torch
import transformers
import os


def prepare_model_input(s):
    a = torch.tensor([[ord(c) for c in s]])
    if torch.cuda.is_available():
        a = a.to("cuda")
    return a


def convert_model_output(model_input, model_output):
    a = model_output[0][len(model_input[0]):]
    return "".join(chr(x) for x in a)


def move_func(board):
    logging.debug(f"generating for: '{board.fen()}'")
    legal_moves = set(m.uci() for m in board.legal_moves)
    model_input_s = "(" + " ".join(m.uci() for m in board.move_stack)
    if model_input_s != "(":
        model_input_s += " "
    model_input = prepare_model_input(model_input_s)
    if model_input.shape[1] > max_input_len:
        logging.debug("fail: input too long")
        return "FAIL_LONG_INPUT"
    
    for i in range(num_attempts):
        # ~ print(model_input)
        model_output = model.generate(model_input, min_new_tokens=5, max_new_tokens=5, do_sample=True)
        # ~ print(model_output)
        model_output_s = convert_model_output(model_input, model_output)
        model_output_s = model_output_s.strip()
        logging.debug(f"attempt {i+1}/{num_attempts}: {model_output_s}")
        if model_output_s in legal_moves:
            logging.debug(f"{model_output_s} legal :)")
            return model_output_s
        else:
            logging.debug(f"{model_output_s} illegal :(")
    logging.debug("fail: no valid attempts")
    return model_output_s


if __name__ == "__main__":
    model_path = os.getenv("GPT2_MODEL_PATH_UCI")
    if model_path is None:
        raise Exception("please specify model path")
    num_attempts = 1
    model = transformers.GPT2LMHeadModel.from_pretrained(model_path)
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    max_input_len = 1000
    
    engine_name = "MyEngine-gpt2-uci"
    engine.enable_logging(engine_name)
    e = engine.Engine(engine_name, move_func)
    e.run()
