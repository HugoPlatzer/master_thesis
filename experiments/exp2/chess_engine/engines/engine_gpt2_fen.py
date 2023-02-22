import engine
import chess
import logging
import torch
import transformers
import os


def get_model_input(board):
    model_input_s = board.fen() + ":"
    if len(model_input_s) > 95:
        raise Exception("input too long")
    model_input = torch.tensor([[ord(c) for c in model_input_s]])
    if torch.cuda.is_available():
        model_input = model_input.to("cuda")
    return model_input


def convert_model_output(model_input, model_output):
    a = model_output[0][len(model_input[0]):]
    return "".join(chr(x) for x in a)


def move_func(board):
    logging.debug(f"generating for: '{board.fen()}'")
    legal_moves = set(m.uci() for m in board.legal_moves)
    model_input = get_model_input(board)
    
    for i in range(num_attempts):
        # ~ print(model_input)
        model_output = model.generate(model_input, min_new_tokens=0, max_new_tokens=10, do_sample=True)
        # ~ print(model_output)
        model_output_s = convert_model_output(model_input, model_output)
        model_output_s = model_output_s[:-1].strip()
        logging.debug(f"attempt {i+1}/{num_attempts}: {model_output_s}")
        if model_output_s in legal_moves:
            logging.debug(f"{model_output_s} legal :)")
            return model_output_s
        else:
            logging.debug(f"{model_output_s} illegal :(")
    logging.debug("fail: no valid attempts")
    return model_output_s


if __name__ == "__main__":
    model_path = os.getenv("GPT2_MODEL_PATH_FEN")
    if model_path is None:
        raise Exception("please specify model path")
    num_attempts = 1
    model = transformers.GPT2LMHeadModel.from_pretrained(model_path)
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    engine_name = "MyEngine-gpt2-fen"
    engine.enable_logging(engine_name)
    e = engine.Engine(engine_name, move_func)
    e.run()
