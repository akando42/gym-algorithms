import gymnasium as gym
import numpy as np
import pickle
import time

with open("pendulum_actor_critic.pkl", "rb") as f:
    policy = pickle.load(f)

actor_w = policy["actor_w"]

def policy(obs):
    x = obs
    mu = np.dot(actor_w, x)            # mean of Gaussian
    action = np.clip(mu, -2.0, 2.0)    # deterministic (mean only)
    return np.array([action], dtype=np.float32)


def metrics(obs):
        cos_theta = obs[0]
        sin_theta = obs[1]
        theta_dot = obs[2]
        theta = np.degrees(np.arctan2(sin_theta, cos_theta))
        # print("Angle:", theta, "Angle Velocity: ", theta_dot)
        return (theta, theta_dot)

env = gym.make("Pendulum-v1", render_mode = "human")
obs, info = env.reset(seed=0)

total_reward = 0

for t in range(200):
    action = policy(obs)

    next_obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    obs = next_obs
    (theta, theta_dot) = metrics(next_obs)
    print("Step ", t, theta, theta_dot, "Reward ", reward)

    if terminated or truncated:
        break

    time.sleep(0.02)  # slow down playback

env.close()
print("Episode reward:", total_reward)