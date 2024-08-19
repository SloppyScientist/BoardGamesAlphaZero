import numpy as np

def get_state_board(fen, flip=False):
	fen = flip_fen(fen, flip)
	planes = np.vstack((position_planes(fen), helper_planes(fen)))
	if flip:
		assert check_current_planes(fen, planes), f"get_state_board: fen is not flipped"
	assert planes.shape == (18, 8, 8), f"get_state_board: shape is {planes.shape} instead of (18,8,8)"
	return planes

def flip_fen(fen, flip):
	if not flip:
		return fen
	foo = fen.split()

	pos = foo[0].split("/")
	foo[0] = "/".join([char.swapcase() for char in reversed(pos)])
	foo[1] = 'w' if foo[1] == 'b' else 'b'
	foo[2] = "".join(sorted([char.swapcase() for char in foo[2]]))

	return " ".join(foo)

def helper_planes(fen):
	foo = fen.split()
	en_passant = np.zeros((8, 8), dtype=np.float32)
	alg_to_coord = lambda col, row: (8-int(row), ord(col)-ord('a')) 
	if foo[3] != '-':
		col, row = foo[3][0], foo[3][1]
		rank, file = alg_to_coord(col, row)
		en_passant[rank][file] = 1

	fifty_move_count = np.full((8, 8), int(foo[4]), dtype=np.float32)

	planes = [np.full((8, 8), int('K' in foo[2]), dtype=np.float32),
				np.full((8, 8), int('Q' in foo[2]), dtype=np.float32),
				np.full((8, 8), int('k' in foo[2]), dtype=np.float32),
				np.full((8, 8), int('q' in foo[2]), dtype=np.float32),
				fifty_move_count, en_passant]
	planes = np.asarray(planes, dtype=np.float32)
	assert planes.shape == (6,8,8), f"helper_planes shape is:{planes.shape} instead of (6,8,8)"
	return planes

def replace_tags_board(fen):
	fen = list(fen.split()[0])
	for idx, char in enumerate(fen):
		fen[idx] = '1' * int(char) if char.isnumeric() else char
	return "".join(fen).replace("/","")

pieces_order = 'KQRBNPkqrbnp'
pieces_dict = {pieces_order[i]: i for i in range(12)}

def position_planes(fen):
	board_state = replace_tags_board(fen)
	pieces_both = np.zeros((12,8,8), dtype=np.float32)

	for rank in range(8):
		for file in range(8):
			piece = board_state[rank * 8 + file]
			if piece.isalpha():
				pieces_both[pieces_dict[piece]][rank][file] = 1

	assert pieces_both.shape == (12,8,8), f"position_planes shape is: {pieces_both.shape} instead of (12,8,8)"
	return pieces_both

castling_order = "KQkq"

def check_current_planes(fen, planes):
	# Recreate positional string
	position_planes = planes[0:12]
	assert position_planes.shape == (12,8,8)
	fakefen = ['1'] * 64
	for i in range(12):
		for rank in range(8):
			for file in range(8):
				if position_planes[i][rank][file] == 1:
					assert fakefen[rank * 8 + file] == '1'
					fakefen[rank * 8 + file] = pieces_order[i]
	fakefen = "".join(fakefen)
	# Recreate castling string
	castling_planes = planes[12:16]
	castling_string = ""
	for i in range(4):
		if castling_planes[i][0][0] == 1:
			castling_string += castling_order[i]
	if len(castling_string) == 0:
		castling_string = '-'
	# Recreate en_passant string
	en_passant_planes = planes[17]
	en_passant_string = '-'
	coord_to_alg = lambda rank, file: chr(ord('a') + file) + str(8 - rank)
	for rank in range(8):
		for file in range(8):
			if en_passant_planes[rank][file] == 1:
				en_passant_string = coord_to_alg(rank, file)
	# Recreate fifty_move_count string
	fifty_move_string = str(int(planes[16][0][0]))

	realparts = fen.split()
	assert realparts[1] == 'w'
	assert realparts[2] == castling_string
	assert realparts[3] == en_passant_string
	assert realparts[4] == fifty_move_string

	return fakefen == replace_tags_board(fen)

def create_uci_labels():
	labels_array = []
	letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
	numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
	promoted_to = ['q', 'r', 'b', 'n']

	for l1 in range(8):
		for n1 in range(8):
			destinations = [(t, n1) for t in range(8)] + \
							[(l1, t) for t in range(8)] + \
							[(l1 + t, n1 + t) for t in range(-7, 8)] + \
							[(l1 + t, n1 - t) for t in range(-7, 8)] + \
							[(l1 + a, n1 + b) for (a, b) in
							[(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]
			for (l2, n2) in destinations:
				if (l1, n1) != (l2, n2) and l2 in range(8) and n2 in range(8):
					move = letters[l1] + numbers[n1] + letters[l2] + numbers[n2]
					labels_array.append(move)
	for l1 in range(8):
		l = letters[l1]
		for p in promoted_to:
			labels_array.append(l + '2' + l + '1' + p)
			labels_array.append(l + '7' + l + '8' + p)
			if l1 > 0:
				l_l = letters[l1 - 1]
				labels_array.append(l + '2' + l_l + '1' + p)
				labels_array.append(l + '7' + l_l + '8' + p)
			if l1 < 7:
				l_r = letters[l1 + 1]
				labels_array.append(l + '2' + l_r + '1' + p)
				labels_array.append(l + '7' + l_r + '8' + p)
	return labels_array

"""
Copyright (c) 2017 Samuel Gravan (part of the code is due to Ken Morishita)

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""