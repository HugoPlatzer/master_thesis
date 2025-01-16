import logging
import chess
import sys
import os


def enable_logging(engine_name):
    base_path = os.getenv("BASE_PATH")
    if base_path == "":
        raise Exception("missing env variable")
    logfile_dir = os.path.join(base_path, "logs")
    logfile_name = os.path.join(logfile_dir, f"log_{engine_name}.txt")
    logging.basicConfig(format='%(filename)s:%(funcName)s %(asctime)s %(message)s',
        filename=logfile_name, level=logging.DEBUG)
    sys.excepthook = log_exception


def log_exception(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


def send_response(line):
    logging.debug(f"SEND: '{line}'")
    print(line)
    sys.stdout.flush()


class Engine:
    def __init__(self, engine_name, move_func):
        # move_func: board -> uci_string
        self.engine_name = engine_name
        self.move_func = move_func
        self.board = None
    
    
    def run(self):
        while True:
            cmd_line = sys.stdin.readline().strip()
            logging.debug(f"READ: '{cmd_line}'")
            cmd_parts = cmd_line.split()
            cmd, cmd_params = cmd_parts[0], cmd_parts[1:]
            
            if cmd == "uci":
                send_response(f"id name {self.engine_name}")
                send_response("uciok")
            elif cmd == "isready":
                send_response("readyok")
            elif cmd == "ucinewgame":
                pass
            elif cmd == "position":
                self.board = chess.Board()
                if cmd_params[0] != "startpos":
                    raise Exception("unknown command structure")
                if len(cmd_params) > 1:
                    if cmd_params[1] != "moves":
                        raise Exception("unknown command structure")
                    moves_uci = cmd_params[2:]
                    logging.debug(f"pushing moves {moves_uci}")
                    for m in moves_uci:
                        self.board.push_uci(m)
            elif cmd == "go":
                legal_moves = list(self.board.legal_moves)
                logging.debug(f"{len(legal_moves)} legal moves")
                if len(legal_moves) == 0:
                    move_to_play = "(none)"
                else:
                    move_to_play = self.move_func(self.board)
                send_response(f"bestmove {move_to_play}")
            elif cmd == "quit":
                exit()
            else:
                send_response("Unknown command")
