import chess
from chess_helper import board_rotation, get_rank_file, adj_mask, adj_pawns


def material_heuristic(board):

    # https://python-chess.readthedocs.io/en/latest/core.html
    # chess.PAWN: chess.PieceType= 1
    # chess.KNIGHT: chess.PieceType= 2
    # chess.BISHOP: chess.PieceType= 3
    # chess.ROOK: chess.PieceType= 4
    # chess.QUEEN: chess.PieceType= 5
    # chess.KING: chess.PieceType= 6

    # Hans Berliner's system
    # https://en.wikipedia.org/wiki/Chess_piece_relative_value

    material_scores = [100, 320, 333, 510, 880, 20000]
    score = 0
    for piece_type in chess.PIECE_TYPES:
        white_player = board.pieces(piece_type, chess.WHITE)
        black_player = board.pieces(piece_type, chess.BLACK)
        score += material_scores[piece_type - 1] * len(white_player) - material_scores[
            piece_type - 1
        ] * len(black_player)
    return score


def pst_heuristic(board):
    # The matrices values was found on chessprogramming
    # https://www.chessprogramming.org/Simplified_Evaluation_Function

    pst_matrix = []
    # pawn
    pst_matrix.append([0,  0,  0,  0,  0,  0,  0,  0,  50, 50, 50, 50, 50, 50, 50, 50,  10, 10, 20, 30, 30, 20, 10, 10,  5,  5, 10, 25, 25, 10,
                       5,  5,  0,  0,  0, 20, 20,  0,  0,  0,  5, -5, -10,  0,  0, -10, -5,  5,  5, 10, 10, -20, -20, 10, 10,  5,  0,  0,  0,  0,  0,  0,  0,  0])
    # knight
    pst_matrix.append([-50, -40, -30, -30, -30, -30, -40, -50, -40, -20,  0,  0,  0,  0, -20, -40, -30,  0, 10, 15, 15, 10,  0, -30, -30,  5, 15, 20, 20, 15,
                       5, -30, -30,  0, 15, 20, 20, 15,  0, -30, -30,  5, 10, 15, 15, 10,  5, -30, -40, -20,  0,  5,  5,  0, -20, -40, -50, -40, -30, -30, -30, -30, -40, -50, ])
    # bishop
    pst_matrix.append([-20, -10, -10, -10, -10, -10, -10, -20, -10,  0,  0,  0,  0,  0,  0, -10, -10,  0,  5, 10, 10,  5,  0, -10, -10,  5,  5, 10, 10,  5,
                       5, -10, -10,  0, 10, 10, 10, 10,  0, -10, -10, 10, 10, 10, 10, 10, 10, -10, -10,  5,  0,  0,  0,  0,  5, -10, -20, -10, -10, -10, -10, -10, -10, -20, ])
    # rook
    pst_matrix.append([0,  0,  0,  0,  0,  0,  0,  0,  5, 10, 10, 10, 10, 10, 10,  5, -5,  0,  0,  0,  0,  0,  0, -5, -5,  0,  0,  0,  0,  0,
                       0, -5, -5,  0,  0,  0,  0,  0,  0, -5, -5,  0,  0,  0,  0,  0,  0, -5, -5,  0,  0,  0,  0,  0,  0, -5,  0,  0,  0,  5,  5,  0,  0,  0])
    # queen
    pst_matrix.append([-20, -10, -10, -5, -5, -10, -10, -20, -10,  0,  0,  0,  0,  0,  0, -10, -10,  0,  5,  5,  5,  5,  0, -10, -5,  0,  5,  5,  5,  5,
                       0, -5,  0,  0,  5,  5,  5,  5,  0, -5, -10,  5,  5,  5,  5,  5,  0, -10, -10,  0,  5,  0,  0,  0,  0, -10, -20, -10, -10, -5, -5, -10, -10, -20])
    # king
    pst_matrix.append([-30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -
                       50, -40, -40, -30, -20, -30, -30, -40, -40, -30, -30, -20, -10, -20, -20, -20, -20, -20, -20, -10, 20, 20,  0,  0,  0,  0, 20, 20, 20, 30, 10,  0,  0, 10, 30, 20])

    score = 0
    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            pos = 7 * (7 - int((square - 1) / 8)) + (square - 1) % 8
            score += pst_matrix[piece - 1][pos]
        for square in board_rotation(board.pieces(piece, chess.BLACK)):
            pos = 7 * (7 - int((square - 1) / 8)) + (square - 1) % 8
            score -= pst_matrix[piece - 1][pos]
    return score


def is_isolated(pawn, pawns):
    return int(adj_pawns(pawn, pawns)) > 0


def is_doubled_isolated(pawn, pawns, opponent_pawns):
    (p_rank, p_file) = get_rank_file(pawn)
    return (
        is_isolated(pawn, pawns)
        and chess.BB_FILES[p_file] & int(opponent_pawns) > 0
        and chess.BB_FILES[p_file] & (~chess.BB_RANKS[p_rank]) & int(pawns) > 0
    )


def is_doubled(pawn, pawns):
    (p_rank, p_file) = get_rank_file(pawn)
    return (
        is_isolated(pawn, pawns)
        and chess.BB_FILES[p_file] & chess.BB_RANKS[p_rank + 1] & int(pawns) > 0
    )


def is_backward(pawn, pawns, opponent_pawns):
    (p_rank, p_file) = get_rank_file(pawn)

    #  Is backward?
    adjp = adj_pawns(pawn, pawns)
    for y in range(p_rank, 7):
        if adjp & chess.BB_RANKS[y] > 0:
            return False

    # Can safely move?
    tmp_mask = 0
    if p_rank > 0:
        tmp_mask = p_file & chess.BB_RANKS[p_rank - 1]
    if p_rank > 1:
        tmp_mask = tmp_mask | (adj_mask(pawn) & chess.BB_RANKS[p_rank - 2])
    if tmp_mask & int(opponent_pawns) > 0:
        return True

    return False


def is_supported(pawn, pawns):
    (p_rank, _) = get_rank_file(pawn)
    return int(adj_pawns(pawn, pawns)) & chess.BB_RANKS[p_rank + 1] > 0


def is_phalanx(pawn, pawns):
    (p_rank, _) = get_rank_file(pawn)
    return int(adj_pawns(pawn, pawns)) & chess.BB_RANKS[p_rank] > 0


def is_connected(pawn, pawns):
    return is_phalanx(pawn, pawns) or is_supported(pawn, pawns)


def is_opposed(pawn, opponent_pawns):
    (_, p_file) = get_rank_file(pawn)
    return chess.BB_FILES[p_file] & int(opponent_pawns) > 0


def connected_bonus(pawn, pawns, opponent_pawns):
    (p_rank, _) = get_rank_file(pawn)
    seed = [0, 7, 8, 12, 29, 48, 86, 0]
    oppo = is_opposed(pawn, opponent_pawns)
    phal = is_phalanx(pawn, pawns)
    supp = is_supported(pawn, pawns)
    return seed[p_rank] * (2 + phal - oppo) + 21 * supp


def is_weak_unopposed(pawn, pawns, opponent_pawns):
    if is_opposed(pawn, opponent_pawns):
        return False
    return is_isolated(pawn, pawns) or is_backward(pawn, pawns, opponent_pawns)


def pawn_score(a, b):
    score = 0
    for pawn in a:
        if is_doubled_isolated(pawn, a, b):
            score -= 11
        elif is_isolated(pawn, a):
            score -= 5
        elif is_backward(pawn, a, b):
            score -= 9

        if is_doubled(pawn, a):
            score -= 11

        if is_connected(pawn, a):
            score += connected_bonus(pawn, a, b)

        if is_weak_unopposed(pawn, a, b):
            score -= 13
    return score


def pawn_heuristic(board):
    WP = board.pieces(chess.PAWN, chess.WHITE)
    BP = board.pieces(chess.PAWN, chess.BLACK)
    return pawn_score(WP, BP) - pawn_score(board_rotation(BP), board_rotation(WP))


def attacked_by(piece_type, pieces):
    attacked_mask = 0
    if piece_type == chess.PAWN:
        for pawn in pieces:
            (p_rank, _) = get_rank_file(pawn)
            if(p_rank > 0):
                attacked_mask = attacked_mask | (
                    adj_mask(pawn) & chess.BB_RANKS[p_rank - 1]
                )
    return attacked_mask


def mobility_area(pawns, king, opponent_pawns):
    return chess.SquareSet(~ (int(pawns) | int(king) | attacked_by(chess.PAWN, opponent_pawns)))


def mobility_heuristic(board):
    bonus = {
        chess.KNIGHT: [-62, -53, -12, -4, 3, 13, 22, 28, 33],
        chess.BISHOP: [-48, -20, 16, 26, 38,
                       51, 55, 63, 63, 68, 81, 81, 91, 98],
        chess.ROOK: [-60, -20, 2, 3, 3, 11,
                     22, 31, 40, 40, 41, 48, 57, 57, 62],
        chess.QUEEN: [-30, -12, -8, -9, 20, 23, 23, 35, 38, 53, 64, 65, 65, 66, 67,
                      67, 72, 72, 77, 79, 93, 108, 108, 108, 110, 114, 114, 116]
    }

    score = 0
    for piece_type in bonus.keys():
        for piece in board.pieces(piece_type, chess.WHITE):
            score += bonus[piece_type][len(
                chess.SquareSet(board.attacks_mask(piece)))]

        for piece in board.pieces(piece_type, chess.BLACK):
            score -= bonus[piece_type][len(
                chess.SquareSet(board.attacks_mask(piece)))]

    pawns = board.pieces(chess.PAWN, chess.WHITE)
    king = board.pieces(chess.QUEEN, chess.WHITE)
    opponent_pawns = board.pieces(chess.PAWN, chess.BLACK)
    score += len(mobility_area(pawns, king, opponent_pawns))

    pawns = board.pieces(chess.PAWN, chess.BLACK)
    king = board.pieces(chess.QUEEN, chess.BLACK)
    opponent_pawns = board.pieces(chess.PAWN, chess.WHITE)
    score -= len(mobility_area(pawns, king, opponent_pawns))

    return score


def evaluation(board):
    return (
        material_heuristic(board)
        + 0.3 * pst_heuristic(board)
        + 0.3 * pawn_heuristic(board)
        + 0.5 * mobility_heuristic(board)
    )


# def get_heuristics(board, color):
#     h1 = material_heuristic(board) * (-1 if not color else 1)
#     h2 = pst_heuristic(board) * (-1 if not color else 1)
#     h3 = pawn_heuristic(board) * (-1 if not color else 1)
#     h4 = mobility_heuristic(board) * (-1 if not color else 1)
def get_heuristics(board):
    h1 = material_heuristic(board)
    h2 = pst_heuristic(board)
    h3 = pawn_heuristic(board)
    h4 = mobility_heuristic(board)
    return (h1, h2, h3, h4)
