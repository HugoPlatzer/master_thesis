import numpy as np
import h5py
import tqdm
import chess

samples_file = "../dataset/games_uci_shuf.txt"
n_games = 1000
sep_char = ":"
pad_char = " "
end_char = ";"
sample_len = 73
out_file = "samples.hdf5"

def get_sample(board, move_str):
    def get_piece(pmap, i):
        try:
            return pmap[i].symbol()
        except KeyError:
            return "."
    
    pmap = board.piece_map()
    piece_str = "".join(get_piece(pmap, i) for i in range(64))
    player_str = "w" if board.turn else "b"
    if len(move_str) == 4:
        move_str += " "
    sample_str = f"{piece_str}:{player_str}:{move_str};"
    if len(sample_str) != sample_len:
        raise Exception("wrong length")
    sample = [ord(c) for c in sample_str]
    return sample
    
    

dset_file = h5py.File(out_file, "w")
dset = dset_file.create_dataset("a", shape=(0, sample_len), maxshape=(None, sample_len), dtype="u1", compression="gzip")

with open(samples_file) as f:
    for i in tqdm.tqdm(range(n_games)):
        game_str = f.readline()
        moves = game_str.split()
        board = chess.Board()
        new_samples = []
        for move in moves:
            sample = get_sample(board, move)
            new_samples.append(sample)
            board.push_uci(move)
        dset.resize((dset.shape[0] + len(new_samples), sample_len))
        dset[-len(new_samples):] = new_samples


print(f"{dset.shape[0]} samples saved")
