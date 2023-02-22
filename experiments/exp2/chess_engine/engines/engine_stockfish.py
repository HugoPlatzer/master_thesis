import engine
import chess
import logging
import subprocess
import os



class StockfishEngine:
    def __init__(self, depth, skill_level):
        stockfish_path = os.getenv("STOCKFISH_PATH")
        if stockfish_path is None:
            raise Exception("please specify stockfish executable path")
        self.depth = depth
        self.skill_level = skill_level
        self.handle = subprocess.Popen(stockfish_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.send_command(f"setoption name Skill Level value {skill_level}")

    def send_command(self, cmd):
        logging.debug(f"SF SEND: '{cmd}'")
        self.handle.stdin.write((cmd + "\n").encode("ascii"))
        self.handle.stdin.flush()
    
    def read_response(self):
        s = self.handle.stdout.readline()
        s = s.decode("ascii").strip()
        logging.debug(f"SF READ: '{s}'")
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
        self.send_command(f"go depth {self.depth}")
        resp = self.read_response()
        while not resp.startswith("bestmove"):
            resp = self.read_response()
        bestmove = resp.split()[1]
        return bestmove


if __name__ == "__main__":
    depth = 1
    skill_level = 0
    
    engine_name = f"MyEngine-stockfish-depth{depth}"
    engine.enable_logging(engine_name)
    
    sf = StockfishEngine(depth, skill_level)
    move_func = lambda board: sf.gen_move(board)
    e = engine.Engine(engine_name, move_func)
    e.run()
