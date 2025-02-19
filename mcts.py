import torch
import numpy as np
from alphazero_nn import AlphaZeroNN
from tic_tac_toe import TicTacToe
from logger import logger

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state  # Tic-Tac-Toe board state
        self.parent = parent
        self.children = {}  # Stores child nodes
        self.visits = 0
        self.value = 0
        self.prior = 0  # Prior probability from the neural network

    def get_best_child(self, exploration_weight=1.4):
        """Selects the best child node using Upper Confidence Bound (UCB1)."""
        def ucb_score(child):
            if child.visits == 0:
                return float('inf')  # Prioritize unexplored moves
            return (child.value / (child.visits + 1)) + exploration_weight * child.prior

        return max(self.children.values(), key=ucb_score)

class MCTS:
    def __init__(self, model, simulations=50):
        self.model = model
        self.simulations = simulations

    def search(self, game):
        logger.debug("🌲 Running MCTS Simulation...")
        root = MCTSNode(game.board.flatten())

        for _ in range(self.simulations):
            node, search_path = root, [root]

            while node.children:
                node = node.get_best_child()
                search_path.append(node)

            # Expansion
            state_tensor = torch.tensor(node.state, dtype=torch.float32).unsqueeze(0)
            policy, value = self.model(state_tensor)
            policy = policy.detach().numpy().flatten()
            value = value.item()

            for move, prob in zip(game.get_valid_moves(), policy):
                game_copy = TicTacToe()
                game_copy.board = node.state.reshape(3, 3).copy()
                game_copy.make_move(move)
                new_state = game_copy.board.flatten()
                node.children[move] = MCTSNode(new_state, parent=node)
                node.children[move].prior = prob

            # Backpropagation
            for node in reversed(search_path):
                node.visits += 1
                node.value += value

        best_move = max(root.children.items(), key=lambda x: x[1].visits)[0]
        logger.info(f"🤖 Best Move Selected by MCTS: {best_move}")
        return best_move