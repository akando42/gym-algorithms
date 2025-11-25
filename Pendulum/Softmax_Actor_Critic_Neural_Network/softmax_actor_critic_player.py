import gymnasium as gym
import numpy as np
import time


# ======================================================
#  LOAD SAVED MODEL
# ======================================================
data = np.load("numpy_softmax_actor_critic_model.npz", allow_pickle=True)

actor = data["actor"].item()         # dict: W1, b1, W2, b2
critic = data["critic"].item()       # unused during play
ACTION_VALUES = data["action_values"]


# ======================================================
#  Neural Network Forward Pass (same as training)
# ======================================================
def forward(model, x):
    z1 = x @ model["W1"] + model["b1"]
    h1 = np.tanh(z1)
    z2 = h1 @ model["W2"] + model["b2"]
    return z1, h1, z2


def softmax(logits):
    c = np.max(logits)
    exp_l = np.exp(logits - c)
    return exp_l / np.sum(exp_l)


# ======================================================
#  Policy (Greedy or Stochastic)
# ======================================================

def select_action(obs, greedy=True):
    x = obs.astype(np.float32)

    z1, h1, logits = forward(actor, x)
    probs = softmax(logits)

    if greedy:
        action_idx = np.argmax(probs)
    else:
        action_idx = np.random.choice(len(ACTION_VALUES), p=probs)

    action_val = float(ACTION_VALUES[action_idx])
    return np.array([action_val], dtype=np.float32), action_val, action_idx


# ======================================================
#  PLAY EPISODE
# ======================================================
def play(greedy=True, render_delay=0.02):
    env = gym.make("Pendulum-v1", render_mode="human")
    obs, info = env.reset(seed=0)

    total_reward = 0.0

    for t in range(300):
        action_array, action_val, action_idx = select_action(obs, greedy=greedy)

        next_obs, reward, terminated, truncated, info = env.step(action_array)
        total_reward += reward

        obs = next_obs

        if terminated or truncated:
            break

        time.sleep(render_delay)

    env.close()
    print(f"Episode Reward = {total_reward:.2f}")


# ======================================================
#  RUN
# ======================================================
if __name__ == "__main__":
    print("Running trained NumPy Softmax Actor–Critic policy...")
    play(greedy=True)     # change to greedy=False for stochastic actions
