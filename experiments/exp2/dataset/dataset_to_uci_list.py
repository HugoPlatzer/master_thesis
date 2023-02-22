import chess
import chess.pgn
import logging
from pathlib import Path
import re
import io
import multiprocessing
import tqdm


def parse_game_uci(game_str):
    # ~ print(repr(game_str))
    str_io = io.StringIO(game_str)
    game = chess.pgn.read_game(str_io)
    board = game.end().board()
    moves = " ".join(m.uci() for m in board.move_stack)
    return moves


def games_iterator(path):
    pgn_files = [str(f) for f in Path(path).rglob("*.pgn")]
    pgn_files.sort()
    game_pattern = r"\n\n([^\[]+)\n\n"
    for f in pgn_files:
        content = open(f).read()
        for game_match in re.finditer(game_pattern, content):
            game_str = game_match.group(1)
            yield game_str


if __name__ == "__main__":
    db_path = "Lichess Elite Database/"
    out_file = "games_uci.txt"
    
    games_iter = games_iterator(db_path)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    uci_strings_iter = pool.imap_unordered(parse_game_uci, games_iter, chunksize=100)
    uci_strings_iter = tqdm.tqdm(uci_strings_iter)
    
    with open(out_file, "w") as f:
        for s in uci_strings_iter:
            print(s, file=f)
    
