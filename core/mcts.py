import torch
import math
import time
import numpy as np


class Node:
    def __init__(self, prior, to_play, action_taken=None):
        self.prior = prior
        self.to_play = to_play
        self.action_taken = action_taken

        self.visit_count = 0
        self.value_sum = 0
        
        self.children = {}
        self.state = None
    
    def expanded(self):
        return len(self.children) > 0
    
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, child, cpuct):
        prior_score = cpuct * child.prior * math.sqrt(self.visit_count) / (child.visit_count + 1)
        if child.visit_count > 0:
            value_score = -child.value()
        else:
            value_score = 0
        return value_score + prior_score

    def select_action(self, temperature):
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]

        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / np.sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)
        
        return action

    def get_action_probs(self, game, temperature):
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        action_probs = np.zeros(game.action_size, dtype=float)
        
        if temperature == 0:
            action = self.select_action(temperature)
            action_probs[action] = 1
        elif temperature == float("inf"):
            for action in actions:
                action_probs[action] = 1 / len(actions)
        else:
            visit_count_distribution = visit_counts ** (1 / temperature)
            if np.sum(visit_count_distribution) == 0:
                for action in actions:
                    action_probs[action] = 1 / len(actions)
            else:
                visit_count_distribution = visit_count_distribution / np.sum(visit_count_distribution)
                for idx, action in enumerate(actions):
                    action_probs[action] = visit_count_distribution[idx]
    
        action_probs[np.isnan(action_probs)] = 0
        action_probs = action_probs / np.sum(action_probs)        
        return action_probs

    def select_child(self, cpuct):
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = self.ucb_score(child, cpuct)
            if score > best_score:
                best_score = score
                best_action = action 
                best_child = child
        
        return best_action, best_child

    def expand(self, state, to_play, action_probs):
        self.to_play = to_play
        self.state = state
        for action, prob in enumerate(action_probs):
            if prob != 0:
                self.children[action] = Node(prior=prob, to_play=self.to_play * -1, action_taken=action)
    
    def __repr__(self):
        return f"{self.state.__str__()} Prior:{self.prior:.2f} Count: {self.visit_count} Value: {self.value()}"

 
class MonteCarloTreeSearch:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
    
    def search(self, state, to_play):
        start_time = time.process_time()
        root = Node(0, to_play)
        action_probs, _ = self._get_prediction(state, dirichlet=True)
        root.expand(state, to_play, action_probs)
        # while time.process_time() - start_time < self.args.time_limit:
        for _ in range(self.args.num_searches):
            node = root
            search_path = [node]

            while node.expanded():
                action, node = node.select_child(self.args.cpuct)
                search_path.append(node)

            parent = search_path[-2]
            state = parent.state

            next_state= self.game.get_next_state(state, action, player=1)
            next_state = self.game.get_canonical_board(next_state, player=-1)

            value = self.game.get_reward(next_state, player=1)
            if value is None:
                action_probs, value = self._get_prediction(next_state)
                node.expand(next_state, parent.to_play*-1, action_probs)
            
            self.backup(search_path, value, parent.to_play*-1)
        return root
    
    @torch.no_grad
    def _get_prediction(self, state, dirichlet=False):
        policy, value = self.model(torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0))
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        if dirichlet:
            policy = (1 - self.args.dirichlet_epsilon) * policy + self.args.dirichlet_epsilon * np.random.dirichlet([self.args.dirichlet_alpha] * self.game.action_size)
        policy = policy * self.game.get_valid_moves(state)
        policy = policy / np.sum(policy)
        value = value.item()
        return policy, value
    
    def backup(self, search_path, value, to_play):
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1