#!/usr/bin/env python3
from GlobalData import GlobalData as gd

import pickle
import chess
import time
import pandas as pd
from ChessPlayer import ChessPlayer
from chessboard import display
from Predictor import Predictor

gd.predictor = Predictor(gd.data_columns)

if not gd.is_first_run:
    gd.load_data()


def play_game(player_white="heuristic", player_black="heuristic"):
    board = chess.Board()
    game_data_frame = pd.DataFrame(columns=gd.data_columns)

    players = [
        ChessPlayer(True, player_white),
        ChessPlayer(False, player_black),
    ]
    game_over = False

    while not game_over:
        for player in players:
            (move, moves_df) = player.get_move(board)
            game_data_frame = game_data_frame.append(
                moves_df, ignore_index=True)
            board.push(move)

            if gd.show_display:
                display.update(board.fen())

            if board.is_game_over():
                game_over = True
                break

    return (board.result(), game_data_frame)


if gd.show_display:
    display.start()

for _ in range(1000):
    (result, game_df) = play_game('ai', 'heuristic')
    gd.results.append(result)

    game_df = game_df.apply(pd.to_numeric, errors='coerce', axis=1)
    gd.games_data = gd.games_data.append(
        game_df, ignore_index=True)

    (result, game_df) = play_game('heuristic', 'ai')
    gd.results.append(result)

    game_df = game_df.apply(pd.to_numeric, errors='coerce', axis=1)
    gd.games_data = gd.games_data.append(
        game_df, ignore_index=True)

    gd.predictor.train_model(gd.games_data)
    gd.save_data()

    print("If you want to stop the training, stop the execution now")
    time.sleep(5)

print(gd.results)

if gd.show_display:
    display.terminate()
