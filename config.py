import argparse

from core.chess_game import ChessGame
from core.tictactoe import TicTacToe
from core.connect import ConnectFour
from core.gomoku import Gomoku

def get_game_and_args():
	parser = argparse.ArgumentParser(description="Pytorch Board Games Training")
	
	parser.add_argument("--env", type=str, default="TicTacToe", help="Name of the game to play")

	parser.add_argument("--cpuct", default=1.0, type=float, help="Exploration constant for ucb score")
	parser.add_argument("--temperature", default=1.0, type=float, help="Starting temperature constant")
	parser.add_argument("--dirichlet_epsilon", default=0.25, type=float, help="Dirichlet equation, epsilon consant, lower means less noise")
	parser.add_argument("--dirichlet_alpha", default=0.3, type=float, help="Dirichlet equation, alpha or eta constant")
	parser.add_argument("--num_searches", default=100, type=int, help="Time given for a mcts search")
	
	parser.add_argument("--datasetID", default=1, type=int,  help="Dataset ID")
	parser.add_argument("--dataset_version", default=1, type=int,  help="Dataset Version")
	parser.add_argument("--training_version", default='A', type=str,  help="Training Version")

	parser.add_argument("--iterations", default=10, type=int,  help="Number of total iterations for main pipeline")
	parser.add_argument("--starting_iteration", default=1, type=int,  help="Starting iteration")
	parser.add_argument("--iteration", default=1, type=int,  help="Current iteration")
	parser.add_argument("--sp_iters", type=int, default=50, help="Number of self play games per iteration")
	parser.add_argument("--workers", default=8, type=int, help="Number of workers to work in parallel")
	parser.add_argument("--first_break", default=5, type=int,  help="Number of iterations before changing parameters for the first time")
	parser.add_argument("--second_break", default=8, type=int,  help="Number of iterations before changing parameters for the second time")
	parser.add_argument("--num_datasets", default=0, type=int,  help="Number of datasets to combine")
	parser.add_argument("--dataset_decay_rate", default=2.0, type=float,  help="Decay ratio of elements of past datasets to combine with new dataset")

	parser.add_argument("--input_channels", default=3, type=int, help="Input channels for Residual Model")
	parser.add_argument("--filters", default=16, type=int, help="Number of filters for Residual Model")
	parser.add_argument("--res_blocks", default=2, type=int, help="Number of blocks for ResTower")

	parser.add_argument("--lr", default=0.001, type=float, help="Learning rate for optimizer")
	parser.add_argument("--scheduler", "-r", action="store_true", help="load scheduler")
	parser.add_argument("--weight_decay", default=0.0001, type=float, help="Weight decay for optimizer")
	parser.add_argument("--batch_size", default=64, type=int, help="Batch size for training")
	parser.add_argument("--num_epochs", default=2, type=int, help="Number of epochs for trainining")

	parser.add_argument("--load_model", "-q", action="store_true", help="load the model")
	parser.add_argument("--parallelize", "-p", action="store_true", help="parallelize selfplay games")
	parser.add_argument("--get_dataset", "-d", action="store_true", help="start selfplay process")
	parser.add_argument("--train_model", "-t", action="store_true", help="start training the model")

	args = parser.parse_args()

	game = get_game(args)

	return game, args

def get_game(args):
	if args.env == "Gomoku":
		game = Gomoku()
	elif args.env == "ConnectFour":
		game = ConnectFour()
	elif args.env == "Chess":
		game = ChessGame()
	else:
		game = TicTacToe()
	return game

def update_args_first_break(args):
	args.cpuct = 2.0
	args.temperature = 1.0
	args.dirichlet_epsilon = 0.25
	args.dirichlet_alpha = 0.6
	args.num_searches = 300

	args.lr = 0.001
	args.batch_size = 128
	args.num_epochs = 4
	args.sp_iters = 25
	args.num_datasets = args.first_break

def update_args_second_break(args):
	args.cpuct = 2.0
	args.temperature = 0.8
	args.dirichlet_epsilon = 0.25
	args.dirichlet_alpha = 0.8
	args.num_searches = 600

	args.lr = 0.001
	args.batch_size = 128
	args.num_epochs = 4
	args.sp_iters = 25
	args.num_datasets = args.second_break