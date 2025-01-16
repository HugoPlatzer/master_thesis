import engine
import chess
import logging
import random


def move_func(board):
    legal_moves = list(board.legal_moves)
    move = random.choice(legal_moves)
    return move.uci()


if __name__ == "__main__":
    engine_name = "MyEngine-random"
    engine.enable_logging(engine_name)
    e = engine.Engine(engine_name, move_func)
    e.run()
