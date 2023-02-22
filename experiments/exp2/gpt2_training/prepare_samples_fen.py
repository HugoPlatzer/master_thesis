import numpy as np
import h5py
import tqdm
import chess
import sys
import multiprocessing



def get_sample(board, move_str):
    fen = board.fen()
    sample_str = f"{fen}:{move_str};"
    if len(sample_str) > sample_len:
        raise Exception("sample too long")
    sample_str += ";" * (sample_len - len(sample_str))
    sample = np.array([ord(c) for c in sample_str], dtype=np.uint8)
    return sample


def samples_from_game(game_str):
    moves = game_str.strip().split()
    board = chess.Board()
    new_samples = []
    for move in moves:
        sample = get_sample(board, move)
        new_samples.append(sample)
        board.push_uci(move)
    if new_samples != []:
        new_samples = np.stack(new_samples)
        return new_samples
    else:
        return None
    

samples_file = sys.argv[1]
out_file = sys.argv[2]
sample_len = 100

dset_file = h5py.File(out_file, "w", driver="stdio")
dset = dset_file.create_dataset("a", shape=(0, sample_len), maxshape=(None, sample_len), dtype="u1", compression="gzip")
pool_size = multiprocessing.cpu_count()
pool = multiprocessing.Pool(pool_size)

samples_iter = pool.imap_unordered(samples_from_game, open(samples_file))
samples_iter = tqdm.tqdm(samples_iter)

for new_samples in samples_iter:
    # ~ print(new_samples.shape)
    if new_samples is not None:
        dset.resize((dset.shape[0] + new_samples.shape[0], sample_len))
        dset[-new_samples.shape[0]:] = new_samples

dset_file.flush()
print(f"{dset.shape[0]} samples saved")
