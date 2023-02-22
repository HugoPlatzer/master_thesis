import numpy as np
import h5py
import tqdm
import chess
import chess.pgn


def get_model_input(game_str):
    moves = game_str.strip().split()
    game = chess.pgn.Game()
    game.add_line([chess.Move.from_uci(m) for m in moves])
    pgn_string = str(game.mainline())
    if not (min_chars <= len(pgn_string) <= max_chars):
        return None
    pgn_string += pad_char * (max_chars - len(pgn_string))
    model_input = [ord(c) for c in pgn_string]
    return model_input


games_file = "../dataset/games_uci_shuf.txt"
# ~ n_samples = 100000
n_games_total = 10000
# ~ n_samples = 3819130
min_chars = 5
max_chars = 1024
pad_char = ";"
out_file = "samples.hdf5"

dset_file = h5py.File(out_file, "w")
dset = dset_file.create_dataset("a", shape=(n_games_total, max_chars), dtype="u1", compression="gzip")

n_games = 0
with open(games_file) as f:
    for i in tqdm.tqdm(range(n_games_total)):
        model_input = get_model_input(f.readline())
        if model_input is not None:
            dset[n_games] = model_input
            n_games += 1

dset.resize((n_games, max_chars))
print(f"{n_games} games")
