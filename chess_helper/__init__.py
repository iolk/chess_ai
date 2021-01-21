from GlobalData import GlobalData as gd

import chess
import time
import pandas as pd


def board_rotation(board):
    return chess.flip_vertical(board)


def get_rank_file(piece):
    return (chess.square_rank(piece), chess.square_file(piece))


def adj_mask(pawn):
    (_, p_file) = get_rank_file(pawn)
    adj_mask = 0
    if p_file < 7:
        adj_mask = chess.BB_FILES[p_file+1] | adj_mask
    if p_file > 0:
        adj_mask = chess.BB_FILES[p_file-1] | adj_mask
    return adj_mask


def adj_pawns(pawn, pawns):
    adj_pawns = adj_mask(pawn) & int(pawns)
    return adj_pawns
