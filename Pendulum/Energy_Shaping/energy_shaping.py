import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# ==============================
# ENERGY SHAPING CONTROLLER
# ==============================
class EnergyShapingController:
    def __init__(self, kp=5.0, kd=1.0, max_torque=2.0):
        self.kp = kp
        self.kd = kd
        self.max_torque = max_torque
        self.m = 1.0   # mass
        self.l = 1.0   # length
        self.g = 9.8   # gravity

        # Desired energy (upright position)
        self.E_desired = self.m * self.g * self.l

    def get_torque(self, obs):
        cos_theta, sin_theta, theta_dot = obs

        # Recover angle [-pi, pi]
        theta = np.arctan2(sin_theta, cos_theta)

        # Compute system energy
        KE = 0.5 * (theta_dot ** 2) * (self.m * (self.l ** 2))
        PE = self.m * self.g * self.l * (1 - cos_theta)
        E = KE + PE

        # Energy difference
        dE = E - self.E_desired

        # Energy-shaping swing-up torque
        u_swing = -1.0 * theta_dot * dE

        # Stabilizing PD controller around upright θ = 0
        u_balance = -self.kp * theta - self.kd * theta_dot

        # Combine both
        tau = u_swing + u_balance

        # Clip to Pendulum action limits
        return np.array([np.clip(tau, -self.max_torque, self.max_torque)])


# ==============================
# TRAINER LOOP FOR EVALUATION
# ==============================
env = gym.make("Pendulum-v1", render_mode="human")
controller = EnergyShapingController()

num_steps = 200
rewards = []
total_reward = 0

obs, info = env.reset(seed=0)

for t in range(num_steps):
    action = controller.get_torque(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    rewards.append(total_reward)

    if terminated or truncated:
        obs, info = env.reset()

env.close()

print("Final Total Reward:", total_reward)

# ==============================
# PLOT REWARD GROWTH
# ==============================
plt.figure(figsize=(8,5))
plt.plot(rewards)
plt.xlabel("Timestep")
plt.ylabel("Cumulative Reward")
plt.title("Energy Shaping Controller – Reward Growth")
plt.grid(True)
plt.savefig("energy_shaping_reward.png", dpi=60)
plt.show()
