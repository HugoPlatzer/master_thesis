import h5py
import tqdm
import multiprocessing
import sys
import numpy as np

samples_file = sys.argv[1]
out_file = sys.argv[2]
min_tokens = 3
max_tokens = 200
pad_char = ")"




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

def parse_game(game_str):
    game_start_token = 32767
    pad_token = 0
    a = [game_start_token] + [pack_move(m) for m in game_str.split()]
    if not min_tokens <= len(a) <= max_tokens:
        return None
    a += [pad_token] * (max_tokens - len(a))
    return np.array(a, dtype=np.uint16)

pool_size = multiprocessing.cpu_count()
pool = multiprocessing.Pool(pool_size)
parsed_games_iter = pool.imap_unordered(parse_game, open(samples_file))
parsed_games_iter = tqdm.tqdm(parsed_games_iter)

games = []
n_total = 0
for parsed_game in parsed_games_iter:
    n_total += 1
    if parsed_game is not None:
        games.append(parsed_game)

dset_file = h5py.File(out_file, "w")
dset = dset_file.create_dataset("a", shape=(len(games), max_tokens), dtype="u2", compression="gzip")
dset[:] = games
print(f"{len(games)}/{n_total} games")
