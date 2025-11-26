import gymnasium as gym
import numpy as np

class PID:
    def __init__(self, kp, ki, kd, dt, out_min=-2.0, out_max=2.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.integral = 0.0
        self.prev_error = 0.0

        self.out_min = out_min
        self.out_max = out_max

    def compute(self, error):
        # Integral term
        self.integral += error * self.dt

        # Derivative term
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        # PID
        u = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Output torque clipping
        u = np.clip(u, self.out_min, self.out_max)
        return np.array([u], dtype=np.float32)


def run_pid_pendulum():
    env = gym.make("Pendulum-v1", render_mode="human")
    obs, _ = env.reset()

    dt = 0.05   # environment time step

    # Tune these by hand:
    pid = PID(
        kp=16.0,   # stronger pull toward upright
        ki=0.0,    # no integral → avoids windup and oscillation
        kd=2,    # strong damping for stability
        dt=dt,
        out_min=-2.0,
        out_max=2.0
    )

    for step in range(300):
        # Extract angle and angular velocity
        cos_th, sin_th, th_dot = obs
        theta = np.arctan2(sin_th, cos_th)

        # Goal = upright (theta = 0)
        error = theta

        # Compute control
        torque = pid.compute(-error - 0.1 * th_dot)   # add damping term

        obs, reward, terminated, truncated, _ = env.step(torque)
        env.render()

        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    run_pid_pendulum()
