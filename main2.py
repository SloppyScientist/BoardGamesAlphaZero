import warnings
warnings.filterwarnings("ignore", ".*Applied workaround*.",)

import torch
import multiprocessing

import numpy as np

from core.mcts import MonteCarloTreeSearch
from core.model import ResNet
from core.agent import AlphaZero

from config import get_game_and_args, update_args_first_break, update_args_second_break

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    np.set_printoptions(suppress=True)
    game, args = get_game_and_args()

    for iteration in range(args.starting_iteration, args.iterations+1, 1):
        print(f"This is iteration {iteration}!")
        args.iteration = iteration
        
        if iteration > args.first_break:
            update_args_first_break(args)
        
        if iteration > args.second_break:
            update_args_second_break(args)

        print("===> Building Model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ResNet(game, device, input_channels=args.input_channels, filters=args.filters, res_blocks=args.res_blocks)
        print(f"...Device: {device}")
        if device == "cuda":
            model = torch.nn.DataParallel(model)
            torch.backends.cudnn.benchmark = True
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if iteration != 1:
            model.load_state_dict(torch.load(f"./model/{repr(game)}_v{args.dataset_version}{args.training_version}_it{args.iteration-1}.pt"))
            optimizer.load_state_dict(torch.load(f"./optim/{repr(game)}_v{args.dataset_version}{args.training_version}_it{args.iteration-1}.pt"))
            print(f"Loaded model {repr(game)}_v{args.dataset_version}{args.training_version}_it{args.iteration-1}.pt")

        mcts = MonteCarloTreeSearch(game, args, model)
        agent = AlphaZero(model, optimizer, game, args, mcts)

        if args.get_dataset:
            agent.create_dataset(parallel=args.parallelize)
        
        if args.train_model:
            dataset = agent.combine_datasets_v2()
            if args.scheduler:
                agent.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr, eta_min=0.001)
            agent.train_dataset(dataset=dataset)