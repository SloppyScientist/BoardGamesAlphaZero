from scipy.signal import convolve2d

import numpy as np

class ConnectFour:
    def __init__(self, rows=6, columns=7):
        self.rows = rows
        self.columns = columns
        self.action_size = columns

    def __repr__(self):
        return "ConnectFour"

    def get_initial_state(self):
        return np.zeros((self.rows, self.columns))

    def get_next_state(self, state, action, player):
        height = np.max(np.where(state[:, action] == 0))
        copied_state = state.copy()
        copied_state[height, action] = player
        return copied_state

    def get_current_player(self, state):
        pos_ones = np.count_nonzero(state == 1)
        neg_ones = np.count_nonzero(state == -1)
        return "BLACK" if pos_ones > neg_ones else "WHITE"

    def get_valid_moves(self, state):
        return (state[0] == 0).astype(np.uint8)

    def get_encoded_state(self, state):
        return np.stack((state==1, state==0, state==-1)).astype(np.float32)

    def get_canonical_board(self, state, player):
        return state * player

    def check_state(self, state):
        # Create kernels for horizontal, vertical and diagonal win detection
        horizontal_kernel = np.array([[1, 1, 1, 1]])
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(4, dtype=np.uint8)
        diag2_kernel = np.fliplr(diag1_kernel)

        detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]

        # Use convolve2d function, 4 indicates there were 4 connected tiles in the board
        for kernel in detection_kernels:
            if (convolve2d(state, kernel, mode="valid") == 4).any():
                    return True
        return False

    def get_reward(self, state, player):
        # Set all of a player's tiles to 1, everything else 0
        current = (state==player).astype(np.float32)
        opponent = (state==-player).astype(np.float32)

        if self.check_state(current):
            return 1
        if self.check_state(opponent):
            return -1
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0
        
        return None
    
    def render(self, state):
        board = state.copy()
        print('=============================')

        for row in range(self.rows):
            for col in range(self.columns):
                print('| {} '.format('X' if board[row][col] == 1 else 'O' if board[row][col] == -1 else ' '), end='')
            print('|')

        print('=============================')