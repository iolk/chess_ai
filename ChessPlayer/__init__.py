from GlobalData import GlobalData as gd

import chess
from heuristics import evaluation, get_heuristics
import numpy as np
import pandas as pd


class ChessPlayer:
    def __init__(self, is_white, player_type):
        self.is_white = is_white
        self.player_type = player_type

    def minimax(self, board, depth, alpha, beta, max_player, use_predictor):
        if depth == 0 or board.is_game_over():
            fen_desc = board.fen()

            if use_predictor and not board.is_game_over():
                (h1, h2, h3, h4) = get_heuristics(board)

                to_predict = gd.predictor.normalize_data(
                    pd.DataFrame([[h1, h2, h3, h4]], columns=gd.input_columns),
                    gd.games_data,
                    gd.to_normalize_cols
                )

                return gd.predictor.predict(to_predict.values.tolist())[0][0]
            else:
                if not (fen_desc in gd.evaluated):
                    gd.evaluated[fen_desc] = evaluation(board)
                    if board.is_game_over():
                        if max_player:
                            gd.evaluated[fen_desc] += -20000
                        else:
                            gd.evaluated[fen_desc] += 20000

                return gd.evaluated[fen_desc]

        if max_player:
            max_eval = -np.inf
            for move in board.legal_moves:
                if move.promotion is not None and move.promotion != 5:
                    continue
                board.push(move)
                eval = self.minimax(
                    board, depth - 1, alpha, beta, False, use_predictor)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = np.inf
            for move in board.legal_moves:
                if move.promotion is not None and move.promotion != 5:
                    continue
                board.push(move)
                eval = self.minimax(
                    board, depth - 1, alpha, beta, True, use_predictor)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def get_heuristic_move(self, board):
        best_move = None
        best_eval = -np.inf
        if not self.is_white:
            best_eval = np.inf

        moves_data = pd.DataFrame(columns=gd.data_columns)

        for move in board.legal_moves:
            board.push(move)

            (h1, h2, h3, h4) = get_heuristics(board)

            is_maximixing = False
            if not self.is_white:
                is_maximixing = True

            fen_desc = board.fen()
            if not (fen_desc in gd.toplevel_evaluated):
                gd.toplevel_evaluated[fen_desc] = self.minimax(
                    board,
                    gd.heuristic_depth - 1,
                    -np.inf,
                    np.inf,
                    is_maximixing,
                    False
                )

            eval = gd.toplevel_evaluated[fen_desc]

            board.pop()

            new_move_data = gd.predictor.normalize_data(
                pd.DataFrame([[h1, h2, h3, h4, eval]],
                             columns=gd.data_columns),
                gd.games_data,
                gd.to_normalize_cols
            )
            moves_data = moves_data.append(new_move_data)

            # print(move, eval)

            if self.is_white:
                if eval > best_eval:
                    best_eval = eval
                    best_move = move
            else:
                if eval < best_eval:
                    best_eval = eval
                    best_move = move

        print("BEST: ", best_move, best_eval)
        return (best_move, moves_data)

    def get_ai_move(self, board):
        best_move = None
        best_eval = -np.inf
        if not self.is_white:
            best_eval = np.inf

        for move in board.legal_moves:
            board.push(move)

            (h1, h2, h3, h4) = get_heuristics(board)

            to_predict = gd.predictor.normalize_data(
                pd.DataFrame([[h1, h2, h3, h4]], columns=gd.input_columns),
                gd.games_data,
                gd.to_normalize_cols
            )

            eval = gd.predictor.predict(to_predict.values.tolist())[0][0]

            board.pop()

            if self.is_white:
                if eval > best_eval:
                    best_eval = eval
                    best_move = move
            else:
                if eval < best_eval:
                    best_eval = eval
                    best_move = move

        return best_move

    def get_ai_minmax_move(self, board):
        best_move = None
        best_eval = -np.inf
        if not self.is_white:
            best_eval = np.inf

        for move in board.legal_moves:
            board.push(move)

            is_maximixing = False
            if not self.is_white:
                is_maximixing = True

            eval = self.minimax(
                board,
                gd.heuristic_depth - 1,
                -np.inf,
                np.inf,
                is_maximixing,
                True
            )

            board.pop()

            if self.is_white:
                if eval > best_eval:
                    best_eval = eval
                    best_move = move
            else:
                if eval < best_eval:
                    best_eval = eval
                    best_move = move

        print("BEST: ", best_move, best_eval)
        return best_move

    def get_move(self, board):
        (move, moves_df) = self.get_heuristic_move(board)

        if self.player_type == "human":
            isValid = False
            while not isValid:
                move = input('Your move: ')
                try:
                    board.push_uci(move)
                    isValid = True
                except:
                    isValid = False

        if self.player_type == "ai":
            move = self.get_ai_move(board)

        if self.player_type == "ai_minmax":
            move = self.get_ai_minmax_move(board)

        return (move, moves_df)
