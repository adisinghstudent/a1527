import torch
import torch.optim as optim
from tic_tac_toe import TicTacToe
from alphazero_nn import AlphaZeroNN
from mcts import MCTS
from logger import logger  # Import logging

def train():
    logger.info("🚀 Starting AlphaZero Training for Tic-Tac-Toe")
    
    model = AlphaZeroNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for episode in range(1000):  # Train for 1000 self-play games
        game = TicTacToe()
        mcts = MCTS(model)
        history = []

        while game.check_winner() is None:
            best_move = mcts.search(game)
            game.make_move(best_move)
            history.append((game.board.flatten(), best_move))

        # Assign rewards based on game outcome
        winner = game.check_winner()
        rewards = [winner if move_idx % 2 == 0 else -winner for move_idx, _ in enumerate(history)]

        # Update model
        total_loss = 0
        for (state, move), reward in zip(history, rewards):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            target_value = torch.tensor([reward], dtype=torch.float32)

            optimizer.zero_grad()
            _, predicted_value = model(state_tensor)
            loss = (target_value - predicted_value).pow(2).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Logging training progress
        logger.info(f"🏆 Episode {episode}: Winner = {winner}, Loss = {total_loss:.4f}")

        if episode % 100 == 0:
            torch.save(model.state_dict(), f"models/alphazero_ep{episode}.pth")
            logger.info(f"💾 Model checkpoint saved at Episode {episode}")

if __name__ == "__main__":
    train()