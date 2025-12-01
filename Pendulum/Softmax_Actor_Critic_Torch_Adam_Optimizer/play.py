import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
import numpy as np

print("Torch OK:", torch.__version__)
print("Numpy OK:", np.__version__)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, actor_lr):
        super(PolicyNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, 64)
        self.fc_2 = nn.Linear(64, 64)
        self.fc_mu = nn.Linear(64, action_dim)
        self.fc_std = nn.Linear(64, action_dim)

        self.lr = actor_lr

        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.max_action = 2
        self.min_action = -2
        self.action_scale = (self.max_action - self.min_action) / 2.0
        self.action_bias = (self.max_action + self.min_action) / 2.0

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = F.leaky_relu(self.fc_1(x))
        x = F.leaky_relu(self.fc_2(x))
        mu = self.fc_mu(x)
        log_std = self.fc_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        reparameter = Normal(mean, std)
        x_t = reparameter.rsample()
        y_t = torch.tanh(x_t)
        action = self.action_scale * y_t + self.action_bias

        # # Enforcing Action Bound
        log_prob = reparameter.log_prob(x_t)
        log_prob = log_prob - torch.sum(torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6), dim=-1, keepdim=True)

        return action, log_prob

### LOAD MODEL
model = PolicyNetwork(3, 1, 0.001).to(
	torch.device("cpu")
)

model.load_state_dict(
	torch.load("models/sac_actor_EP30.pt", map_location="cpu")
)

model.eval()


# RUN IN GYMNASIUM
env = gym.make("Pendulum-v1", render_mode="human")
state, info = env.reset()

done = False
while not done:
    s = torch.tensor(state, dtype=torch.float32).reshape(1, 3)

    with torch.no_grad():
        action, log_prob = model.sample(s)

    # Scale to Pendulum action space
    action = 2.0 * action
    action = float(action.detach().cpu().squeeze())
    print("ACTION ", action)
    next_state, reward, terminated, truncated, info = env.step([action])

    state = next_state
    done = terminated or truncated

env.close()