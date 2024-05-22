import numpy as np
import torch
from torch.distributions import Normal
import matplotlib.pyplot as plt
from toy_experiments.toy_helpers import Data_Sampler
from matplotlib import rcParams
import os

def generate_data(num, radius=2, std=0.02, device="cpu"):
    each_num = int(num / 8)
    angles = np.linspace(0, 2 * np.pi, 9)[:-1]
    means = np.vstack([radius * np.cos(angles), radius * np.sin(angles)]).T

    data_samples = []
    for mean in means:
        distribution = Normal(
            torch.tensor(mean, dtype=torch.float32), torch.tensor([std, std])
        )
        samples = distribution.sample((each_num,)).clip(
            -2.5, 2.5
        ) 
        data_samples.append(samples)

    data = torch.cat(data_samples, dim=0)
    action = data
    state = torch.zeros_like(action)
    reward = torch.zeros((num, 1))
    return Data_Sampler(state, action, reward, device)

device = "cuda:6"

num_data = 10000
data_sampler = generate_data(num_data, device=device)

fig = plt.figure(figsize=(8, 8)) 

rcParams["font.family"] = "Times New Roman"

num_eval = 1000
_, action_samples, _ = data_sampler.sample(num_eval)
action_samples = action_samples.cpu().numpy()

plt.scatter(action_samples[:, 0], action_samples[:, 1], c='#299D8F', alpha=0.25)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.xlabel("action dim 1", fontsize=20)
plt.ylabel("action dim 2", fontsize=20)
plt.title("Ground Truth", fontsize=25)

fig.savefig('Target Multimodal.eps') 
fig.savefig('Target Multimodal.png') 

state_dim = 2
action_dim = 2
max_action = 2.
discount = 0.99
tau = 0.005
hidden_dim = 128

from toy_experiments.bc_diffusion import BC as Diffusion_Agent

num_epochs = 20000
lr = 3e-4
batch_size = 100
iterations = int(num_data / batch_size)

beta_schedule = "vp"
model_type = "MLP"
T = 50

diffusion_agent = Diffusion_Agent(
    state_dim=state_dim,
    action_dim=action_dim,
    max_action=max_action,
    device=device,
    discount=discount,
    tau=tau,
    beta_schedule=beta_schedule,
    n_timesteps=T,
    model_type=model_type,
    hidden_dim=hidden_dim,
    lr=lr,
)

for i in range(num_epochs):

    diffusion_agent.train(data_sampler, iterations=iterations, batch_size=batch_size)

    if i % 100 == 0:
        print(f"Epoch: {i}")

        new_state = torch.zeros((num_eval, 2), device=device)
        new_action = diffusion_agent.actor.sample(new_state)
        new_action = new_action.detach().cpu().numpy()

        fig = plt.figure(figsize=(8, 8))

        plt.scatter(new_action[:, 0], new_action[:, 1], c="#299D8F", alpha=0.25)
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.xlabel("action dim 1", fontsize=20)
        plt.ylabel("action dim 2", fontsize=20)
        plt.title(f"Diffusion Policy at Epoch {i}", fontsize=25)

        os.makedirs("example", exist_ok=True)

        fig.savefig(f"example/Diffusion_Policy_{i}.eps")  
        fig.savefig(f"example/Diffusion_Policy_{i}.png")

        plt.close(fig) 
