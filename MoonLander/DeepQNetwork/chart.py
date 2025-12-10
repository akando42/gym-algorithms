import numpy as np
import matplotlib.pyplot as plt 

scores = np.load("Training_Score.npy")

# print(len(data))
print(scores)

plt.figure(figsize=(6, 4))
plt.plot(scores, label="Training Epoc Score")
plt.xlabel("Epoc")
plt.ylabel("Average Score")
plt.title("DeepQNetwork Moon Lander Training")
plt.savefig("Optima_Agent_Training.png", dpi=100)
plt.grid(True)