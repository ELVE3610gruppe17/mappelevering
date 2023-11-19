import chess
from chess import Move
import chess.engine

def get_computer_move(board):
    stockfish_path = r"C:\Users\ilars\Downloads\stockfish-windows-x86-64-modern\stockfish\stockfish-windows-x86-64-modern.exe"
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        result = engine.play(board, chess.engine.Limit(time=1.0))
        return result.move

board = chess.Board()

while not board.is_game_over():
    print(board)
    
    # Get user input for the move
    user_move_uci = input("Enter your move (in UCI notation): ")
    
    # Check if the user's move is valid
    user_move = Move.from_uci(user_move_uci)
    if user_move in board.legal_moves:
        board.push(user_move)
    else:
        print("Invalid move. Please try again.")
        continue

    # Check for game over after the user's move
    if board.is_game_over():
        break

    # Get the computer's move
    computer_move = get_computer_move(board)
    board.push(computer_move)
    print(f"Computer's move: {computer_move.uci()}")
    
print("Game over. Result: " + board.result())

