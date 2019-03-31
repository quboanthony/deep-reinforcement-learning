import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed,fc1_size=128,fc2_size=64,fc3_size=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.fc1=nn.Linear(state_size,fc1_size)
        self.fc2=nn.Linear(fc1_size,fc2_size)
        self.fc3=nn.Linear(fc2_size,fc3_size)
        self.fc4=nn.Linear(fc3_size,action_size)
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x=F.relu(self.fc1(state))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        return self.fc4(x)
