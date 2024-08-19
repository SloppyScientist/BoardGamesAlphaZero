import numpy as np
import chess

from utils import get_state_board, flip_fen, create_uci_labels

class ChessGame:
    def __init__(self):
        self.rows = 8
        self.columns = 8
        self.board = chess.Board()

        self.action_size = len(create_uci_labels())
        self.total_moves = create_uci_labels()

        self.winner = None

    def __repr__(self):
        return "Chess"

    def get_initial_state(self):
        return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def get_next_state(self, state, action, player):
        self.board.set_fen(state)
        self.board.push(chess.Move.from_uci(self.total_moves[action]))
        return self.board.fen()

    def get_current_player(self, state):
        return "BLACK" if state.split()[1] == 'b' else "WHITE"

    def get_valid_moves(self, state):
        self.board.set_fen(state)
        curr_actions = list(map(str, list(self.board.legal_moves)))
        return np.array([1 if obj in curr_actions else 0 for obj in self.total_moves])

    def get_encoded_state(self, state):
        return get_state_board(state, False)

    def get_canonical_board(self, state, player): 
        return flip_fen(state, True) if player == -1 else state

    def get_reward(self, state, player):
        self.board.set_fen(state)
        result = self.board.result(claim_draw=True)

        if result == "1-0":
            self.winner = "WHITE" if player == 1 else "BLACK"
            return 1
        if result == "0-1":
            self.winner = "BLACK" if player == 1 else "WHITE"
            return -1
        if result == "1/2-1/2":
            self.winner = "DRAW"
            return 0
        return None
    
    def render(self, state):
        self.board.set_fen(state)
        print("----------------")
        print(self.board)
        print("----------------")
        # print(list(map(str, list(self.board.legal_moves))))