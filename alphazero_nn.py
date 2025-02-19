import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaZeroNN(nn.Module):
    def __init__(self):
        super(AlphaZeroNN, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 64)
        self.policy_head = nn.Linear(64, 9)  # Policy: probabilities for 9 positions
        self.value_head = nn.Linear(64, 1)  # Value: game outcome prediction

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = F.softmax(self.policy_head(x), dim=-1)
        value = torch.tanh(self.value_head(x))
        return policy, value

# Example usage:
if __name__ == "__main__":
    net = AlphaZeroNN()
    dummy_state = torch.randn(1, 9)  # Dummy Tic-Tac-Toe board
    policy, value = net(dummy_state)
    print("Policy:", policy.detach().numpy())
    print("Value:", value.item())