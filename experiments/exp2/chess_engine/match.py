import subprocess
import chess
import chess.pgn
import os
import sys
import logging
from datetime import datetime
import multiprocessing
import tqdm


class Engine:
    def __init__(self, executable_path):
        self.executable_path = executable_path
        self.name = executable_path
        self.handle = subprocess.Popen(executable_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def send_command(self, cmd):
        logging.debug(f"engine SEND: '{cmd}'")
        self.handle.stdin.write((cmd + "\n").encode("ascii"))
        self.handle.stdin.flush()
    
    def read_response(self):
        s = self.handle.stdout.readline()
        s = s.decode("ascii").strip()
        logging.debug(f"engine READ: '{s}'")
        return s
    
    def gen_move(self, board):
        played_moves = list(board.move_stack)
        if len(played_moves) > 0:
            cmd = "position startpos moves"
            for m in played_moves:
                cmd += f" {m.uci()}"
            self.send_command(cmd)
        else:
            self.send_command("position startpos")
        self.send_command(f"go")
        resp = self.read_response()
        while not resp.startswith("bestmove"):
            resp = self.read_response()
        bestmove = resp.split()[1]
        return bestmove


class GameResult:
    def __init__(self, moves, result, reason):
        self.moves = moves
        self.result = result
        self.reason = reason


def play_game(engine_black, engine_white, max_plies=400):
    board = chess.Board()
    ply_count = 0
    while ply_count < max_plies and board.result() == "*":
        side_to_move = "black" if ply_count % 2 == 1 else "white"
        engine = engine_black if side_to_move == "black" else engine_white
        move = engine.gen_move(board)
        try:
            board.push_uci(move)
        except Exception:
            result = "1-0" if side_to_move == "black" else "0-1"
            return GameResult(board.move_stack, result, f"Illegal move: '{move}'")
        ply_count += 1
    if board.result == "*":
        result = "1/2-1/2"
    else:
        result = board.result()
    return GameResult(board.move_stack, result, None)


def play_game_for_round(round_nr):
    if round_nr % 2 == 1:
        engine_black = Engine(engine_a_path)
        engine_white = Engine(engine_b_path)
    else:
        engine_black = Engine(engine_b_path)
        engine_white = Engine(engine_a_path)
    game_result = play_game(engine_black, engine_white)
    engine_black.send_command("quit")
    engine_white.send_command("quit")
    return game_result


def generate_pgn(game_result, timestamp, round_nr):
    if round_nr % 2 == 1:
        black_name = engine_a_path
        white_name = engine_b_path
    else:
        black_name = engine_b_path
        white_name = engine_a_path
    headers = {
        "Date": timestamp.strftime("%Y.%m.%d"),
        "Round": round_nr,
        "Black": black_name,
        "White": white_name,
        "Result": game_result.result
    }
    if game_result.reason is not None:
        headers["Comment"] = game_result.reason
    game_pgn = chess.pgn.Game(headers)
    game_pgn.add_line(game_result.moves)
    return str(game_pgn)


def generate_match_info(match_timestamp, results):
    lines = []
    lines.append(f"{engine_a_path} vs. {engine_b_path}")
    lines.append(match_timestamp.strftime("%Y %b %d %H:%M:%S"))
    lines.append(f"{len(results)} games")
    num_wins_a = sum(1 for r in results if r[0] == "a")
    num_wins_b = sum(1 for r in results if r[0] == "b")
    num_draws = sum(1 for r in results if r == "1/2-1/2")
    num_wins_black = sum(1 for r in results if "black" in r)
    num_wins_white = sum(1 for r in results if "white" in r)
    lines.append(f"A vs. B: +{num_wins_a} -{num_wins_b} ={num_draws}")
    lines.append(f"Black wins: {num_wins_black} White wins: {num_wins_white} Draws: {num_draws}")
    return "\n".join(lines)


def get_result_aorb(result, round_nr):
    if result == "1/2-1/2":
        return "1/2-1/2"
    elif result == "1-0" and round_nr % 2 == 1:
        return "b:white"
    elif result == "1-0" and round_nr % 2 == 0:
        return "a:white"
    elif result == "0-1" and round_nr % 2 == 1:
        return "a:black"
    elif result == "0-1" and round_nr % 2 == 0:
        return "b:black"
    else:
        raise Exception("unknown result")



if __name__ == "__main__":
    engine_a_path = sys.argv[1]
    engine_b_path = sys.argv[2]
    num_games = int(sys.argv[3])
    save_dir = "matches/"
    pool_size = os.getenv("POOL_SIZE")
    if pool_size is not None:
        pool_size = int(pool_size)
    else:
        pool_size = 1
    
    match_timestamp = datetime.now()
    match_save_dir = match_timestamp.strftime("%Y-%m-%d-%H%M%S")
    match_save_dir = os.path.join(save_dir, match_save_dir)
    os.mkdir(match_save_dir)
    os.mkdir(os.path.join(match_save_dir, "games"))
    
    pool = multiprocessing.Pool(pool_size)
    
    game_results = list(tqdm.tqdm(pool.imap(play_game_for_round, range(1, num_games + 1)), total=num_games))
    
    stats = []
    for i, game_result in enumerate(game_results):
        round_nr = i + 1
        stats.append(get_result_aorb(game_result.result, round_nr))
        pgn = generate_pgn(game_result, match_timestamp, round_nr)
        pgn_filename = "{:03}.pgn".format(round_nr)
        pgn_path = os.path.join(match_save_dir, "games", pgn_filename)
        print(pgn, file=open(pgn_path, "w"))
    
    match_info = generate_match_info(match_timestamp, stats)
    match_info_path = os.path.join(match_save_dir, "info.txt")
    print(match_info, file=open(match_info_path, "w"))
    
    print(f"Saved to {match_save_dir}")
