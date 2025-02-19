import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)  # 0 = empty, 1 = X, -1 = O
        self.current_player = 1  # X always starts

    def reset(self):
        self.board.fill(0)
        self.current_player = 1
        return self.board.copy()

    def get_valid_moves(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def make_move(self, move):
        i, j = move
        if self.board[i, j] == 0:
            self.board[i, j] = self.current_player
            self.current_player *= -1  # Switch turns
            return True
        return False  # Invalid move

    def check_winner(self):
        for line in np.vstack([self.board, self.board.T, np.diag(self.board), np.diag(np.fliplr(self.board))]):
            if np.abs(line.sum()) == 3:
                return np.sign(line.sum())  # 1 for X, -1 for O
        if not self.get_valid_moves():
            return 0  # Draw
        return None  # Game ongoing

    def render(self):
        symbols = {1: 'X', -1: 'O', 0: '.'}
        for row in self.board:
            print(' '.join(symbols[cell] for cell in row))
        print()