import chess
import chess.engine

stockfish_path = r"C:\Users\ilars\Downloads\stockfish-windows-x86-64-modern(1)\stockfish\stockfish-windows-x86-64-modern.exe"
board_ids = [[1,2,3,0,5,3,4,8], [7,8,9,6,7,4,9,5]] # Placeholder verdier, "1" og "8" indikerer tårn, "5" indikerer konge. Første bracket er rad 1
incomplete_fen = "r3kb1r/pppbqppp/2np1n2/4p3/4P3/2N2N2/PPPPBPPP/R1BQ1RK1"


def can_I_castle(brett_id):
    if brett_id[0][4] != 5:
        C_rights = "-"
    elif brett_id[0][0] != 1 and board_ids[0][4] == 5 and board_ids[0][7] == 8:
        C_rights = "k"
    elif brett_id[0][7] != 8 and board_ids[0][4] == 5 and board_ids[0][0] == 1:
        C_rights = "q"
    elif brett_id[0][7] == 8 and board_ids[0][4] == 5 and board_ids[0][0] == 1:
        C_rights = "kq"
    else:
        C_rights = "-" 
    print(f"Castling rights: {C_rights}")
    return C_rights

def can_I_castle_from_k(brett_id):
    if brett_id[0][4] != 5:
        C_rights = "-"
    elif brett_id[0][0] != 1 and board_ids[0][4] == 5 and board_ids[0][7] == 8:
        C_rights = "k"
    else:
        C_rights = "-" 
    print(f"Castling rights: {C_rights}")
    return C_rights

def can_I_castle_from_q(brett_id):
    if brett_id[0][4] != 5:
        C_rights = "-"
    elif brett_id[0][7] != 8 and board_ids[0][4] == 5 and board_ids[0][0] == 1:
        C_rights = "q"
    else:
        C_rights = "-" 
    print(f"Castling rights: {C_rights}")
    return C_rights

def get_board(uferdig_fen, rokade_rettigheter): 
    Incomplete_FEN = uferdig_fen
    Whose_turn = "b"
    En_passant_rights = "-"
    Half_moves_wo_pawn = 0
    Complete_turns = 1

    Complete_FEN = f"{Incomplete_FEN} {Whose_turn} {rokade_rettigheter} {En_passant_rights} {Half_moves_wo_pawn} {Complete_turns}"

    board = chess.Board(Complete_FEN)
    return board, Complete_FEN

def get_computer_move(board): # For stockfish path, download stockfish 16, unpack, add "r" before the string, copy path to the folder, and add "\stockfish-windows-x86-64-modern.exe"
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        result = engine.play(board, chess.engine.Limit(time=1.0))
        computer_move_algebraic = board.san(result.move)
        board.push(result.move)
        print(board)
        UCI_Algebraic = f"{result.move} {computer_move_algebraic}"
        return UCI_Algebraic

Castling_rights = can_I_castle(board_ids)
Board, Complete_FEN = get_board(incomplete_fen, Castling_rights)
Move_UCI_Algebraic = get_computer_move(Board)
print(f"\nUCI / Algebraic\n{Move_UCI_Algebraic}")
print(f"Complete FEN-code {Complete_FEN}")

# Ny runde

incomplete_fen = "r3kb1r/pppbqppp/2np1n2/4p3/4P3/2N2N2/PPPPBPPP/R1BQ1RK1"
board_ids = [[1,2,3,0,5,3,4,0], [7,8,9,6,7,4,9,5]] # Placeholder verdier, "1" og "8" indikerer tårn, "5" indikerer konge. Første bracket er rad 1

if Castling_rights == "kq": 
    Castling_rights = can_I_castle(board_ids)
elif Castling_rights == "k":
    Castling_rights = can_I_castle_from_k(board_ids)
elif Castling_rights == "q":
    Castling_rights = can_I_castle_from_q(board_ids)
elif Castling_rights == "-":
    Castling_rights = "-"

Board, Complete_FEN = get_board (incomplete_fen, Castling_rights)
Move_UCI_Algebraic = get_computer_move(Board)
print(f"\nUCI / Algebraic\n{Move_UCI_Algebraic}")
print(f"Complete FEN-code {Complete_FEN}")

# Ny runde

incomplete_fen = "r3kb1r/pppbqppp/2np1n2/4p3/4P3/2N2N2/PPPPBPPP/R1BQ1RK1"
board_ids = [[1,2,3,0,0,3,4,0], [7,8,9,6,7,4,9,5]] # Placeholder verdier, "1" og "8" indikerer tårn, "5" indikerer konge. Første bracket er rad 1


if Castling_rights == "kq": 
    Castling_rights = can_I_castle(board_ids)
elif Castling_rights == "k":
    Castling_rights = can_I_castle_from_k(board_ids)
elif Castling_rights == "q":
    Castling_rights = can_I_castle_from_q(board_ids)
elif Castling_rights == "-":
    Castling_rights = "-"

Board, Complete_FEN = get_board (incomplete_fen, Castling_rights)
Move_UCI_Algebraic = get_computer_move(Board)
print(f"\nUCI / Algebraic\n{Move_UCI_Algebraic}")
print(f"Complete FEN-code {Complete_FEN}")

# Ny runde

incomplete_fen = "r3kb1r/pppbqppp/2np1n2/4p3/4P3/2N2N2/PPPPBPPP/R1BQ1RK1"
board_ids = [[1,2,3,0,0,3,4,0], [7,8,9,6,7,4,9,5]] # Placeholder verdier, "1" og "8" indikerer tårn, "5" indikerer konge. Første bracket er rad 1


if Castling_rights == "kq": 
    Castling_rights = can_I_castle(board_ids)
elif Castling_rights == "k":
    Castling_rights = can_I_castle_from_k(board_ids)
elif Castling_rights == "q":
    Castling_rights = can_I_castle_from_q(board_ids)
elif Castling_rights == "-":
    Castling_rights = "-"

Board, Complete_FEN = get_board (incomplete_fen, Castling_rights)
Move_UCI_Algebraic = get_computer_move(Board)
print(f"\nUCI / Algebraic\n{Move_UCI_Algebraic}")
print(f"Complete FEN-code {Complete_FEN}")