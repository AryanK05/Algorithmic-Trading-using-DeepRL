import copy
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class DDDQNTrainer:
    class QNetwork(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc4 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.state_value = nn.Linear(hidden_dim // 2, 1)
            self.advantage = nn.Linear(hidden_dim // 2, output_dim)

        def forward(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            hs = F.relu(self.fc3(h))
            ha = F.relu(self.fc4(h))
            state_val = self.state_value(hs)
            adv = self.advantage(ha)
            adv_mean = adv.mean(dim=1, keepdim=True)
            return state_val + (adv - adv_mean)

    def __init__(self, env, hidden_size=100, gamma=0.97,
                 memory_size=200, batch_size=50,
                 train_freq=10, update_q_freq=20,
                 epsilon_start=1.0, epsilon_min=0.1,
                 epsilon_decay=1e-3, start_reduce_epsilon=200,
                 epoch_num=100):
        self.env = env
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.update_q_freq = update_q_freq
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.start_reduce_epsilon = start_reduce_epsilon
        self.epoch_num = epoch_num
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sample_obs = self.env.reset()
        self.input_dim = len(sample_obs)
        self.action_dim = 3

        self.Q = self.QNetwork(self.input_dim, hidden_size, self.action_dim).to(self.device)
        self.Q_target = copy.deepcopy(self.Q).to(self.device)
        self.optimizer = optim.Adam(self.Q.parameters())
        self.memory = []
        self.total_step = 0

        self.total_rewards = []
        self.total_losses = []
        self.total_profits = []

    def train(self):
        step_max = len(self.env.data) - 1
        start_time = time.time()

        for epoch in range(self.epoch_num):
            obs = self.env.reset()
            total_reward = 0
            total_loss = 0
            done = False
            step = 0

            while not done and step < step_max:
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(self.action_dim)
                else:
                    state_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        q_out = self.Q(state_tensor)
                    action = q_out.argmax(dim=1).item()

                step_result = self.env.step(action)
                if len(step_result) == 4:
                    next_obs, reward, done, info = step_result
                else:
                    next_obs, reward, done = step_result
                    info = {}

                self.memory.append((obs, action, reward, next_obs, done))
                if len(self.memory) > self.memory_size:
                    self.memory.pop(0)

                if len(self.memory) == self.memory_size and self.total_step % self.train_freq == 0:
                    batch = random.sample(self.memory, self.batch_size)
                    bs, ba, br, bnext, bd = zip(*batch)

                    bs = torch.tensor(bs, dtype=torch.float32, device=self.device)
                    ba = torch.tensor(ba, dtype=torch.long, device=self.device)
                    br = torch.tensor(br, dtype=torch.float32, device=self.device)
                    bnext = torch.tensor(bnext, dtype=torch.float32, device=self.device)
                    bd = torch.tensor(bd, dtype=torch.float32, device=self.device)

                    q_values = self.Q(bs)
                    q_val = q_values.gather(1, ba.unsqueeze(1)).squeeze(1)

                    with torch.no_grad():
                        next_q = self.Q(bnext)
                        next_actions = next_q.argmax(dim=1, keepdim=True)
                        q_target_next = self.Q_target(bnext)
                        q_target_val = q_target_next.gather(1, next_actions).squeeze(1)
                        target = br + self.gamma * q_target_val * (1 - bd)

                    loss = F.mse_loss(q_val, target)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

                if self.total_step % self.update_q_freq == 0:
                    self.Q_target.load_state_dict(self.Q.state_dict())

                if self.total_step > self.start_reduce_epsilon:
                    self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

                obs = next_obs
                total_reward += reward
                step += 1
                self.total_step += 1

            self.total_rewards.append(total_reward)
            self.total_losses.append(total_loss)
            self.total_profits.append(getattr(self.env, 'profits', 0))

            print(f"Epoch {epoch+1:3d} | Epsilon {self.epsilon:.3f} | Steps {self.total_step:5d} | "
                  f"Return {total_reward:.4f} | Loss {total_loss:.4f} | Profit {self.total_profits[-1]:.4f}")

            if (epoch + 1) % 5 == 0:
                avg_r = np.mean(self.total_rewards[-5:])
                avg_l = np.mean(self.total_losses[-5:])
                elapsed = time.time() - start_time
                print(f"---- Last 5 episodes Avg Return {avg_r:.4f} | Avg Loss {avg_l:.4f} | Time {elapsed:.1f}s ----")
                start_time = time.time()

        return self.Q

    def test(self, test_env, render=False):
        obs = test_env.reset()
        done = False
        total_reward = 0
        total_profit = 0
        actions = []

        while not done:
            state_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_out = self.Q(state_tensor)
            action = q_out.argmax(dim=1).item()
            actions.append(action)
            result = test_env.step(action)
            if len(result) == 4:
                next_obs, reward, done, info = result
            else:
                next_obs, reward, done = result
            total_reward += reward
            obs = next_obs
            if render and hasattr(test_env, "render"):
                test_env.render()

        total_profit = getattr(test_env, 'profits', 0)

        print(f"Test completed | Total Reward: {total_reward:.4f} | Total Profit: {total_profit:.4f}")
        return total_reward, total_profit, actions


    def plot_metrics(self):
        plt.figure(figsize=(14, 4))

        plt.subplot(1, 3, 1)
        plt.plot(self.total_rewards, label='Reward')
        plt.title('Episode Return')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.grid()

        plt.subplot(1, 3, 2)
        plt.plot(self.total_losses, label='Loss', color='orange')
        plt.title('Episode Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.grid()

        plt.subplot(1, 3, 3)
        plt.plot(self.total_profits, label='Profit', color='green')
        plt.title('Episode Profit')
        plt.xlabel('Episode')
        plt.ylabel('Profit')
        plt.grid()

        plt.tight_layout()
        plt.show()
