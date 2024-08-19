import numpy as np
import pickle
import random
import time
import os

import concurrent.futures

import torch
import torch.nn.functional as F

from tqdm import trange

class AlphaZero:
    def __init__(self, model, optimizer, game, args, mcts, scheduler=None):
        self.model = model
        self.optimizer = optimizer

        self.game = game
        self.args = args

        self.mcts = mcts
        self.scheduler = scheduler
    
    def _experience_buffer(self, memory, value):
        experience_buffer = []

        for state, policy in memory:
            experience_buffer.append((self.game.get_encoded_state(state), policy, value))
        return experience_buffer
    
    def selfplay(self, _):
        memory = []
        move_count = 0

        state = self.game.get_initial_state()
        player = 1
        while True:
            reward = self.game.get_reward(state, player=1)

            if move_count == 80:
                memory.append((state, action_probs))
                return self._experience_buffer(memory, -0.5)

            if reward is not None:
                memory.append((state, action_probs))
                return self._experience_buffer(memory, reward)

            root = self.mcts.search(state, player)
            
            if move_count < 30:
                action_probs = root.get_action_probs(self.game, self.args.temperature)
            else:
                action_probs = root.get_action_probs(self.game, temperature=0)
            move_count += 1

            memory.append((state, action_probs))

            action = np.random.choice(self.game.action_size, p=action_probs)
            state = self.game.get_next_state(state, action, player)
            player = player * -1

    def create_dataset(self, parallel=False):
        print("===> Creating Dataset...")
        start_time = time.time()
        dataset = []
        self.model.eval()
        if parallel:
            for _ in trange(self.args.sp_iters):
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.args.workers) as executor:
                    futures = [executor.submit(self.selfplay, gameID) for gameID in range(1, 8+1)]
                    results = [future.result() for future in concurrent.futures.as_completed(futures)]
                for result in results:
                    dataset += result
        else:
            for sp_iteration in trange(self.args.sp_iters):
                dataset += self.selfplay(sp_iteration+1)
    
        random.shuffle(dataset)
        
        if not os.path.isdir('datasets'):
            os.mkdir('datasets')
        with open(f'datasets/{repr(self.game)}_v{self.args.dataset_version}{self.args.training_version}_it{self.args.iteration}_num{self.args.datasetID}.pkl', 'wb') as f:
            pickle.dump(dataset, f)
        print(f"{repr(self.game)}_v{self.args.dataset_version}{self.args.training_version}_it{self.args.iteration}_num{self.args.datasetID}.pkl created in {(time.time() - start_time) / 60:.2f} minutes...")

    def train(self, dataset):
        values, policies, losses, disparity, uniformity = [], [], [], [], [] 
        chunks = (len(dataset) - 1) // self.args.batch_size + 1
        for i in range(chunks-1):
            sample = dataset[i*self.args.batch_size:(i+1)*self.args.batch_size]
            states, policy_targets, value_targets = zip(*sample)

            states = np.array(states, dtype=np.float32) 
            policy_targets = np.array(policy_targets, dtype=np.float32) 
            value_targets = np.array(value_targets, dtype=np.float32).reshape(-1, 1)

            states = torch.tensor(states, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(states)

            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            values.append(value_loss.item())
            policies.append(policy_loss.item())
            losses.append(loss.item())
            
            target_predictions = policy_targets.squeeze(0).cpu().detach().numpy()

            out_policy = torch.softmax(out_policy, axis=1).squeeze(0).cpu().detach().numpy()
            out_policy = out_policy * target_predictions
            out_policy = out_policy / target_predictions
            out_policy = np.nan_to_num(out_policy)
            out_policy = out_policy / np.sum(out_policy, axis=1, keepdims=True)
            out_policy = torch.tensor(out_policy, dtype=torch.float32, device=self.model.device)

            model_predictions = out_policy.squeeze(0).cpu().detach().numpy()
            
            mp_without_zeros = model_predictions[model_predictions != 0]
            tp_without_zeros = target_predictions[target_predictions != 0]
            
            uniformity += list(np.isclose(mp_without_zeros, tp_without_zeros, rtol=0, atol=0.05).astype('float64'))
            disparity += list(np.abs(mp_without_zeros - tp_without_zeros))

            average_value_loss = sum(values) / len(values)
            average_policy_loss = sum(policies) / len(policies)
            average_total_loss = sum(losses) / len(losses)

        print(f"Percentage of policy values that are close by a margin of 0.05: {sum(uniformity)/len(uniformity) * 100:.1f}%")
        print(f"Average disparity between target policy and model prediction: {sum(disparity) / len(disparity):.3f}")
        print(f"Average value loss: {average_value_loss} / Average policy loss: {average_policy_loss} / Average loss: {average_total_loss}")
    
    def train_dataset(self, dataset):
        print("==> Start training...")
        self.model.train()
        for epoch in trange(self.args.num_epochs):
            print(f"Epoch number {epoch} ---------------------------------------------------------------------------")
            self.train(dataset)
            if self.scheduler is not None:
                self.scheduler.step()
        print("==> Saving Model...")
        if not os.path.isdir("model"):
            os.mkdir("model")
        torch.save(self.model.state_dict(), f"model/{repr(self.game)}_v{self.args.dataset_version}{self.args.training_version}_it{self.args.iteration}.pt")
        if not os.path.isdir("optim"):
            os.mkdir("optim")
        torch.save(self.optimizer.state_dict(),  f"optim/{repr(self.game)}_v{self.args.dataset_version}{self.args.training_version}_it{self.args.iteration}.pt")

    def combine_dataset(self, datasetIDs):
        dataset = []
        for datasetID in datasetIDs:
            with open(f'datasets/{repr(self.game)}{datasetID}_{self.args.dataset_version}.pkl', 'rb') as f:
                print(f"Open {repr(self.game)}{datasetID}_{self.args.dataset_version}.pkl")
                dataset += pickle.load(f)
        print(f"Datasets combined into singe dataset of size: {len(dataset)}")
        return dataset
    
    def combine_datasets_v2(self):
        dataset = []
        decay_factor = 1
        for iteration in range(self.args.iteration, self.args.num_datasets, -1):
            with open(f'datasets/{repr(self.game)}_v{self.args.dataset_version}{self.args.training_version}_it{iteration}_num{self.args.datasetID}.pkl', 'rb') as f:

                print(f"Open {repr(self.game)}_v{self.args.dataset_version}{self.args.training_version}_it{iteration}_num{self.args.datasetID}.pkl")
                current_dataset = pickle.load(f)
                print(f"Dataset has length: {len(current_dataset)} and {len(current_dataset)//decay_factor} elements are added.")
                dataset += random.sample(current_dataset, int(len(current_dataset) // decay_factor))
            decay_factor = decay_factor * self.args.dataset_decay_rate
        print(f"Datasets combined into singe dataset of size: {len(dataset)}")
        return dataset