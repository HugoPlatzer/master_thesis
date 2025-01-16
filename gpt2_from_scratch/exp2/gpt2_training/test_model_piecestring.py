from transformers import GPT2LMHeadModel
import torch
import sys
import chess


def get_model_input(fen_string):
    def get_piece(pmap, i):
        try:
            return pmap[i].symbol()
        except KeyError:
            return "."
    
    board = chess.Board(fen_string)
    pmap = board.piece_map()
    piece_str = "".join(get_piece(pmap, i) for i in range(64))
    player_str = "w" if board.turn else "b"
    query_str = f"{piece_str}:{player_str}:"
    model_input = torch.tensor([[ord(c) for c in query_str]])
    return model_input


def convert_model_output(model_input, model_output):
    a = model_output[0][len(model_input[0]):]
    out_str = "".join(chr(x) for x in a)
    return out_str


model_path = sys.argv[1]
model_input_s = sys.argv[2]

model = GPT2LMHeadModel.from_pretrained(model_path)
model_input = get_model_input(model_input_s)
model_output = model.generate(model_input, min_new_tokens=0, max_new_tokens=10, do_sample=True)
model_output_s = convert_model_output(model_input, model_output)
print(repr(model_output_s))
