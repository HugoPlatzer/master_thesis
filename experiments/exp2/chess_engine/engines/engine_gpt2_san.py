import engine
import chess
import chess.pgn
import logging
import torch
import transformers
import os


def prepare_model_input(board):
    game = chess.pgn.Game()
    game.add_line(board.move_stack)
    san_string = str(game.mainline()).strip()
    if board.turn:
        move_nr = 1 + (board.ply() // 2)
        if san_string != "":
            san_string += " "
        san_string += f"{move_nr}. "
    else:
        san_string += " "
    # ~ print(repr(san_string))
    model_input = torch.tensor([[ord(c) for c in san_string]])
    if torch.cuda.is_available():
        model_input = model_input.to("cuda")
    return model_input


def convert_model_output(model_input, model_output):
    a = model_output[0][len(model_input[0]):]
    return "".join(chr(x) for x in a)


def move_func(board):
    logging.debug(f"generating for: '{board.fen()}'")
    model_input = prepare_model_input(board)
    if model_input.shape[1] > max_input_len:
        logging.debug("fail: input too long")
        return "FAIL_LONG_INPUT"
    
    test_board = board.copy()
    for i in range(num_attempts):
        # ~ print(model_input)
        model_output = model.generate(model_input, min_new_tokens=8, max_new_tokens=8, do_sample=True)
        # ~ print(model_output)
        model_output_s = convert_model_output(model_input, model_output)
        model_output_s = model_output_s.strip().split(" ")[0]
        logging.debug(f"attempt {i+1}/{num_attempts}: {model_output_s}")
        try:
            move_pushed = test_board.push_san(model_output_s)
            logging.debug(f"{model_output_s} legal :)")
            return move_pushed.uci()
        except (chess.InvalidMoveError, chess.IllegalMoveError):
            logging.debug(f"{model_output_s} illegal :(")
    logging.debug("fail: no valid attempts")
    return f"san:'{model_output_s}'"


if __name__ == "__main__":
    model_path = os.getenv("GPT2_MODEL_PATH_SAN")
    if model_path is None:
        raise Exception("please specify model path")
    num_attempts = 1
    model = transformers.GPT2LMHeadModel.from_pretrained(model_path)
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    max_input_len = 1000
    
    engine_name = "MyEngine-gpt2-san"
    engine.enable_logging(engine_name)
    e = engine.Engine(engine_name, move_func)
    e.run()
