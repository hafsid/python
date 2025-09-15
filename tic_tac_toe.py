import random
import math
from colorama import init, Fore, Style

init(autoreset=True)

HUMAN = None
AI = None

def print_board(board):
    print()
    for row in [board[i * 3:(i + 1) * 3] for i in range(3)]:
        print(" | ".join(row))
        if row != board[6:]:
            print("-" * 9)
    print()

def available_moves(board):
    return [i for i, spot in enumerate(board) if spot == " "]

def winner(board):
    win_combos = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
        [0, 4, 8], [2, 4, 6]              # diagonals
    ]
    for combo in win_combos:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] != " ":
            return board[combo[0]]
    if " " not in board:
        return "Tie"
    return None

def minimax(board, depth, is_maximizing):
    result = winner(board)
    if result == AI:
        return 1
    elif result == HUMAN:
        return -1
    elif result == "Tie":
        return 0

    if is_maximizing:
        best_score = -math.inf
        for move in available_moves(board):
            board[move] = AI
            score = minimax(board, depth + 1, False)
            board[move] = " "
            best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for move in available_moves(board):
            board[move] = HUMAN
            score = minimax(board, depth + 1, True)
            board[move] = " "
            best_score = min(score, best_score)
        return best_score

def ai_move(board):
    best_score = -math.inf
    move = None
    for i in available_moves(board):
        board[i] = AI
        score = minimax(board, 0, False)
        board[i] = " "
        if score > best_score:
            best_score = score
            move = i
    return move

def play():
    global HUMAN, AI
    board = [" " for _ in range(9)]

    HUMAN = input("Choose your marker (X/O): ").upper()
    AI = "O" if HUMAN == "X" else "X"

    turn = input("Do you want to go first? (y/n): ").lower()
    game_over = False

    print_board(board)

    while not game_over:
        if turn == "y":
            move = int(input("Enter your move (1-9): ")) - 1
            if board[move] == " ":
                board[move] = HUMAN
                turn = "n"
            else:
                print(Fore.RED + "Invalid move! Try again.")
