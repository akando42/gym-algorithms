import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt


# ========================================================
# UTILITY FUNCTIONS
# ========================================================

def get_features(obs):
    """
    Simple feature vector for Actor & Critic.
    You may expand these later (RBF, tile coding, 2-layer NN, etc.)
    """
    return obs  # 3 features: [cosθ, sinθ, θ_dot]


# ========================================================
# ACTOR–CRITIC AGENT WITH GAUSSIAN POLICY
# ========================================================

class ActorCritic:
    def __init__(self, state_dim, alpha_actor=0.01, alpha_critic=0.1,
                 init_sigma=0.5):
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic

        # Linear policy: μ = wᵀx
        self.actor_w = np.zeros(state_dim)

        # Learnable log standard deviation
        self.log_sigma = np.log(init_sigma)

        # Critic: V(s) = vᵀx
        self.critic_w = np.zeros(state_dim)

    def policy(self, obs):
        x = get_features(obs)
        mu = np.dot(self.actor_w, x)
        sigma = np.exp(self.log_sigma)

        # Sample from Gaussian
        action = mu + sigma * np.random.randn()
        action = np.clip(action, -2.0, 2.0)

        return float(action), mu, sigma

    def update(self, obs, reward, next_obs, done, gamma=0.99):
        x = get_features(obs)
        x2 = get_features(next_obs)

        V = np.dot(self.critic_w, x)
        V2 = np.dot(self.critic_w, x2)

        td_target = reward + gamma * V2 * (1 - done)
        td_error = td_target - V

        # ----- Critic update -----
        self.critic_w += self.alpha_critic * td_error * x

        # ----- Actor update -----
        action, mu, sigma = self.policy(obs)

        # Log-likelihood grad for Gaussian policy
        grad_mu = (action - mu) / (sigma**2)
        grad_log_sigma = ((action - mu)**2 / sigma**2) - 1.0

        self.actor_w += self.alpha_actor * td_error * grad_mu * x
        self.log_sigma += self.alpha_actor * td_error * grad_log_sigma

        return td_error


# ========================================================
# TRAINING LOOP
# ========================================================

env = gym.make("Pendulum-v1")
obs, _ = env.reset(seed=0)

agent = ActorCritic(state_dim=3)

num_episodes = 100
rewards_per_episode = []

for episode in range(num_episodes):
    obs, _ = env.reset()
    total_reward = 0

    for step in range(200):
        action, mu, sigma = agent.policy(obs)

        next_obs, reward, terminated, truncated, info = env.step(
            np.array([action], dtype=np.float32)
        )
        
        done = terminated or truncated

        agent.update(obs, reward, next_obs, done)

        total_reward += reward
        obs = next_obs

        if done:
            break

    rewards_per_episode.append(total_reward)
    print(f"Episode {episode+1}/{num_episodes}, Total Reward = {total_reward:.2f}")


# ========================================================
# SAVE POLICY TO PICKLE
# ========================================================

policy_data = {
    "actor_w": agent.actor_w,
    "log_sigma": agent.log_sigma,
    "critic_w": agent.critic_w,
}

with open("pendulum_actor_critic.pkl", "wb") as f:
    pickle.dump(policy_data, f)

print("\nSaved learned policy to pendulum_actor_critic.pkl")


# ========================================================
# PLOT TRAINING CURVE
# ========================================================

plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Actor–Critic Learning Curve (Pendulum-v1)")
plt.grid(True)
plt.savefig("actor_critic_training_curve.png", dpi=300)
plt.show()
