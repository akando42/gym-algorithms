import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# ======================================================
#  Discretized Action Space for Pendulum-v1
# ======================================================
NUM_ACTIONS = 11
ACTION_VALUES = np.linspace(-2.0, 2.0, NUM_ACTIONS, dtype=np.float32)


# ======================================================
#  Neural Network Utilities (NumPy)
# ======================================================

def init_network(input_dim, hidden_dim, output_dim):
    """Xavier initialization for 2-layer NN."""
    W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
    b1 = np.zeros(hidden_dim)

    W2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
    b2 = np.zeros(output_dim)

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward(model, x):
    """
    Forward pass: returns (z1, h1, z2)
    z1 = xW1 + b1
    h1 = tanh(z1)
    z2 = h1W2 + b2
    """
    z1 = x @ model["W1"] + model["b1"]
    h1 = np.tanh(z1)
    z2 = h1 @ model["W2"] + model["b2"]
    return z1, h1, z2


def softmax(logits):
    c = np.max(logits)
    exp_l = np.exp(logits - c)
    return exp_l / np.sum(exp_l)


# ======================================================
#  Backprop Actor (Softmax Policy)
# ======================================================

def backward_actor(model, x, z1, h1, logits, action_idx, advantage, lr):
    """
    Actor update:
    ∇θ log π(a|s) * advantage
    """
    probs = softmax(logits)

    # d(log π) wrt logits
    dlogits = probs.copy()
    dlogits[action_idx] -= 1
    dlogits *= advantage  # TD error as advantage

    # layer 2 gradients
    dW2 = np.outer(h1, dlogits)
    db2 = dlogits

    # backprop into hidden
    dh1 = (model["W2"] @ dlogits).reshape(-1)          # (hidden_dim,)
    dz1 = dh1 * (1 - np.tanh(z1)**2)                  # (hidden_dim,)

    # layer 1
    dW1 = np.outer(x, dz1)
    db1 = dz1

    # update
    model["W1"] -= lr * dW1
    model["b1"] -= lr * db1
    model["W2"] -= lr * dW2
    model["b2"] -= lr * db2


# ======================================================
#  Backprop Critic (Value Function)
# ======================================================

def backward_critic(model, x, z1, h1, value_pred, td_target, lr):
    """
    Critic update: minimize (V(s) − target)^2
    """
    td_err = value_pred - td_target  # scalar

    # dLoss/dV = td_err
    dV = td_err

    # layer 2
    dW2 = np.outer(h1, np.array([dV]))
    db2 = np.array([dV])

    # backprop to hidden
    dh1 = (model["W2"] @ np.array([dV])).reshape(-1)
    dz1 = dh1 * (1 - np.tanh(z1)**2)

    dW1 = np.outer(x, dz1)
    db1 = dz1

    # update
    model["W1"] -= lr * dW1
    model["b1"] -= lr * db1
    model["W2"] -= lr * dW2
    model["b2"] -= lr * db2


# ======================================================
#  MAIN TRAINING LOOP
# ======================================================
env = gym.make("Pendulum-v1")

# Networks
actor = init_network(input_dim=3, hidden_dim=64, output_dim=NUM_ACTIONS)
critic = init_network(input_dim=3, hidden_dim=64, output_dim=1)

lr_actor = 1e-3
lr_critic = 3e-3
gamma = 0.99

EPISODES = 300
MAX_STEPS = 200

returns = []

for ep in range(EPISODES):
    obs, info = env.reset()
    total_reward = 0.0

    for step in range(MAX_STEPS):

        x = obs.astype(np.float32)

        # ---- ACTOR forward ----
        z1_a, h1_a, logits = forward(actor, x)
        probs = softmax(logits)
        action_idx = np.random.choice(NUM_ACTIONS, p=probs)
        action_val = ACTION_VALUES[action_idx]

        action = np.array([action_val], dtype=np.float32)

        # ---- STEP ENV ----
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward

        # ---- CRITIC forward ----
        z1_c, h1_c, value_pred = forward(critic, x)
        value_pred = value_pred.item()

        if done:
            td_target = reward
        else:
            _, _, next_value_pred = forward(critic, next_obs.astype(np.float32))
            td_target = reward + gamma * next_value_pred.item()

        td_error = td_target - value_pred

        # ---- UPDATE CRITIC ----
        backward_critic(
            critic,
            x, z1_c, h1_c,
            value_pred=value_pred,
            td_target=td_target,
            lr=lr_critic
        )

        # ---- UPDATE ACTOR ----
        backward_actor(
            actor,
            x, z1_a, h1_a, logits,
            action_idx=action_idx,
            advantage=td_error,
            lr=lr_actor
        )

        obs = next_obs
        if done:
            break

    returns.append(total_reward)
    print(f"Episode {ep+1}/{EPISODES}  Return = {total_reward:.2f}")

env.close()

# =====================================================
#  Save the trained model
# =====================================================
np.savez(
    "numpy_softmax_actor_critic_model.npz",
    actor=np.array(actor, dtype=object),     # FIXED
    critic=np.array(critic, dtype=object),   # FIXED
    action_values=ACTION_VALUES              # OK
)
print("\nSaved model to numpy_softmax_actor_critic_model.npz")

# =====================================================
#  Plot Learning Curve
# =====================================================
plt.plot(returns)
plt.xlabel("Episode")
plt.ylabel("Total Return")
plt.title("NumPy Softmax Actor–Critic on Pendulum-v1")
plt.grid(True)
plt.savefig("numpy_softmax_actor_critic_learning.png", dpi=300)
plt.show()



