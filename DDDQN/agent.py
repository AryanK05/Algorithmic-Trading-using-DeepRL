import copy
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

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

    def __init__(self, env, hidden_size=128, gamma=0.97,
                 memory_size=200, batch_size=64,
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



class DDDQNTrainer_PrioritizedReplay:
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
        
    class PrioritizedReplayBuffer:
        def __init__(self, max_len, min_len, offset= 0.1):
            self.max_len = max_len
            self.min_len = min_len
            self.offset = offset

            self.buffer = deque(maxlen=max_len)
            self.priorities = deque(maxlen=max_len)

        def is_min_len_reached(self):
            return len(self.buffer) >= self.min_len

        def add(self, transition):
            if self.priorities:
                max_prio = max(self.priorities)
            else:
                max_prio = self.offset
            self.buffer.append(transition)
            self.priorities.append(max_prio)

        def update_priorities(self, indices, errors):
            for idx, err in zip(indices, errors):
                prio = abs(err) + self.offset
                # Ensure idx in range
                if 0 <= idx < len(self.priorities):
                    self.priorities[idx] = prio

        def probabilities(self, priority_scale):
            prios = np.array(self.priorities, dtype=np.float64)
            scaled = prios ** priority_scale
            total = scaled.sum()
            if total == 0:
                # avoid division by zero
                return np.ones_like(scaled) / len(scaled)
            return scaled / total

        def importance_weights(self, priority_scale, importance_weight_scaling, indices=None):
            probs = self.probabilities(priority_scale)
            N = len(probs)
            if indices is None:
                idxs = np.arange(N)
            else:
                idxs = np.array(indices, dtype=int)
            weights = (1.0 / (N * probs[idxs])) ** importance_weight_scaling

            max_w = weights.max() if len(weights) > 0 else 1.0
            return weights / max_w

        def sample(self, batch_size, priority_scale, importance_weight_scaling):
            if not self.is_min_len_reached():
                raise ValueError(f"Not enough samples to sample: {len(self.buffer)}/{self.min_len}")

            probs = self.probabilities(priority_scale)
            N = len(probs)
            indices = np.random.choice(N, size=batch_size, replace=False, p=probs)
            transitions = [self.buffer[idx] for idx in indices]
            is_weights = self.importance_weights(priority_scale, importance_weight_scaling, indices)
            return transitions, indices.tolist(), is_weights

    def __init__(self, env, hidden_size=128, gamma=0.97,
                 mem_min_len = 400, mem_max_len = 1_000,
                 batch_size=64, train_freq=10, train_times = 5, update_q_freq=20,
                 epsilon_start=1.0, epsilon_min=0.1,
                 epsilon_decay=0.95, start_reduce_epsilon=5,
                 epoch_num=100):
        self.env = env
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.train_times = train_times
        self.update_q_freq = update_q_freq
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.start_reduce_epsilon = start_reduce_epsilon
        self.epoch_num = epoch_num
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sample_obs = self.env.reset()
        self.input_dim = len(sample_obs)
        self.action_dim = self.env.action_space

        self.Q = self.QNetwork(self.input_dim, hidden_size, self.action_dim).to(self.device)
        self.Q_target = copy.deepcopy(self.Q).to(self.device)
        self.optimizer = optim.Adam(self.Q.parameters())
        

        
        self.memory = self.PrioritizedReplayBuffer(mem_max_len, mem_min_len)
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
                # select action
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(self.action_dim)
                else:
                    state_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        q_out = self.Q(state_tensor)
                    action = q_out.argmax(dim=1).item()

                # step environment; unpack info if provided
                step_result = self.env.step(action)
                if len(step_result) == 4:
                    next_obs, reward, done, info = step_result
                else:
                    next_obs, reward, done = step_result
                    info = {}

                # store in memory
                self.memory.add((obs, action, reward, next_obs, done))
        
                
                if self.memory.is_min_len_reached() and self.total_step % self.train_freq == 0:
                    for i in range(self.train_times):
                        transition, indices, is_weights = self.memory.sample(self.batch_size, 0.7, 1-self.epsilon)
                        
                        bs, ba, br, bnext, bd = zip(*transition)
                        
                        # print(bs)
                        bs = torch.tensor(bs, dtype=torch.float32, device=self.device)
                        
        
                        
                        ba = torch.tensor(ba, dtype=torch.long, device=self.device)
                        br = torch.tensor(br, dtype=torch.float32, device=self.device)
                        bnext = np.array(bnext)
                        bnext = torch.tensor(bnext, dtype=torch.float32, device=self.device)
                        bd = torch.tensor(bd, dtype=torch.float32, device=self.device)
                        is_weights = torch.tensor(is_weights, dtype=torch.float32, device=self.device)
                        

                        q_values = self.Q(bs)
                        
                        q_val = q_values.gather(1, ba.unsqueeze(1)).squeeze(1)
        
                        # Double DQN target
                        with torch.no_grad():
                            next_q = self.Q(bnext)
                            next_actions = next_q.argmax(dim=1, keepdim=True)
                            q_target_next = self.Q_target(bnext)
                            q_target_val = q_target_next.gather(1, next_actions).squeeze(1)
                            target = br + self.gamma * q_target_val * (1 - bd)
        
                        errors = (target - q_val).detach().cpu().numpy()
                        
                        loss = ((target-q_val) ** 2) * is_weights
                        loss = loss.mean()
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        total_loss += loss.item()

                        
                        self.memory.update_priorities(indices, errors)
        
                        # Debuggin to check if reward is not normalized somehow
                        # if abs(reward) > 10:
                        #     print(f"env_step: {getattr(self.env, 'current_step', 0)}  reward: {reward}    loss: {loss.item()}")

                # update target network
                if self.total_step % self.update_q_freq == 0:
                    self.Q_target.load_state_dict(self.Q.state_dict())

                # epsilon decay
                if self.total_step > self.start_reduce_epsilon:
                    self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
                
                obs = next_obs
                total_reward += reward
                step += 1
                self.total_step += 1

            # end of episode logging
            self.total_rewards.append(total_reward)
            self.total_losses.append(total_loss)
            self.total_profits.append(getattr(self.env, 'total_profit', 0))
            # episode return == total_reward, profit info in env.profits
            print(f"Epoch {epoch+1:3d} | Epsilon {self.epsilon:.3f} | Steps {self.total_step:5d} | Return {total_reward:.4f} | Loss {total_loss:.4f} | Profit {getattr(self.env, 'total_profit', 0):.4f}")
            if hasattr(self.env, "render_confusion_matrix"):
                self.env.render_confusion_matrix()
            

            # periodic summary every 5 episodes
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
        if hasattr(test_env, "render_confusion_matrix"):
                test_env.render_confusion_matrix()
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



       
class DDDQNTrainer_PrioritizedReplay_BiLSTM:
    '''
    Note this traininer can currently only interface with the Day trading env

    Please instantiate the Day trading env with observation_dim argument as '2D' for this network
    '''
    def build_mlp(self, input_dim, hidden_dims, activation=nn.ReLU):
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(activation())
            last_dim = h
        return nn.Sequential(*layers)


    class DuelingBiLSTM(nn.Module):
        def __init__(
            self,
            single_day_dim: int,
            look_back_window: int,
            action_space_dim: int,
            rnn_hidden_space: tuple,
            fc_hidden_space: tuple
        ):
            super().__init__()
            
            # Build stacked bidirectional LSTMs
            self.rnns = nn.ModuleList()
            input_dim = single_day_dim
            for hidden_dim in rnn_hidden_space:
                self.rnns.append(
                    nn.LSTM(
                        input_size=input_dim,
                        hidden_size=hidden_dim,
                        num_layers=1,
                        bidirectional=True,
                        batch_first=True
                    )
                    
                )
                # after averaging forward/backward, next input_dim is hidden_dim
                input_dim = hidden_dim
            
            # Final encoded dimension (after averaging)
            self.encoded_dim = rnn_hidden_space[-1]

            # Build fully connected stack
            self.mlp = self.build_mlp(self.encoded_dim, fc_hidden_space)
            last_fc_dim = fc_hidden_space[-1] if fc_hidden_space else self.encoded_dim

            # Dueling heads
            self.value_head = nn.Linear(last_fc_dim, 1)
            self.advantage_head = nn.Linear(last_fc_dim, action_space_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            x: tensor of shape (batch, look_back_window, single_day_dim)
            returns: tensor of shape (batch, action_space_dim) -- Q-values
            """
            out = x
            # Pass through stacked bi-LSTMs with averaging of forward/back outputs
            for rnn in self.rnns:
                rnn_out, _ = rnn(out)
                # rnn_out shape: (batch, seq_len, 2*hidden_dim)
                h = rnn_out.size(2) // 2
                # average forward and backward features
                out = (rnn_out[:, :, :h] + rnn_out[:, :, h:]) / 2
            # out shape: (batch, seq_len, encoded_dim)

            # take last time step
            last_output = out[:, -1, :]  # (batch, encoded_dim)

            # Pass through FC layers
            features = self.mlp(last_output)

            # Compute value and advantage
            value = self.value_head(features)            # (batch, 1)
            advantage = self.advantage_head(features)    # (batch, action_space_dim)

            # Normalize advantage
            advantage_mean = advantage.mean(dim=1, keepdim=True)
            advantage_normalized = advantage - advantage_mean

            # Combine value and normalized advantage
            q_vals = value + advantage_normalized       # (batch, action_space_dim)
            return q_vals
        
    class PrioritizedReplayBuffer:
        def __init__(self, max_len, min_len, offset= 0.1):
            self.max_len = max_len
            self.min_len = min_len
            self.offset = offset

            self.buffer = deque(maxlen=max_len)
            self.priorities = deque(maxlen=max_len)

        def is_min_len_reached(self):
            return len(self.buffer) >= self.min_len

        def add(self, transition):
            if self.priorities:
                max_prio = max(self.priorities)
            else:
                max_prio = self.offset
            self.buffer.append(transition)
            self.priorities.append(max_prio)

        def update_priorities(self, indices, errors):
            for idx, err in zip(indices, errors):
                prio = abs(err) + self.offset
                # Ensure idx in range
                if 0 <= idx < len(self.priorities):
                    self.priorities[idx] = prio

        def probabilities(self, priority_scale):
            prios = np.array(self.priorities, dtype=np.float64)
            scaled = prios ** priority_scale
            total = scaled.sum()
            if total == 0:
                # avoid division by zero
                return np.ones_like(scaled) / len(scaled)
            return scaled / total

        def importance_weights(self, priority_scale, importance_weight_scaling, indices=None):
            probs = self.probabilities(priority_scale)
            N = len(probs)
            if indices is None:
                idxs = np.arange(N)
            else:
                idxs = np.array(indices, dtype=int)
            weights = (1.0 / (N * probs[idxs])) ** importance_weight_scaling

            max_w = weights.max() if len(weights) > 0 else 1.0
            return weights / max_w

        def sample(self, batch_size, priority_scale, importance_weight_scaling):
            if not self.is_min_len_reached():
                raise ValueError(f"Not enough samples to sample: {len(self.buffer)}/{self.min_len}")

            probs = self.probabilities(priority_scale)
            N = len(probs)
            indices = np.random.choice(N, size=batch_size, replace=False, p=probs)
            transitions = [self.buffer[idx] for idx in indices]
            is_weights = self.importance_weights(priority_scale, importance_weight_scaling, indices)
            return transitions, indices.tolist(), is_weights

    def __init__(self, env, 
                 rnn_hidden_size=128, fcc_hidden_state=128,
                 gamma=0.97,
                 mem_min_len = 400, mem_max_len = 1_000,
                 batch_size=64, train_freq=10, train_times = 5, update_q_freq=20,
                 epsilon_start=1.0, epsilon_min=0.1,
                 epsilon_decay=0.95, start_reduce_epsilon=5,
                 epoch_num=60):
        self.env = env
        self.rnn_hidden_size = rnn_hidden_size
        self.fcc_hidden_size = fcc_hidden_state
        self.gamma = gamma
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.train_times = train_times
        self.update_q_freq = update_q_freq
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.start_reduce_epsilon = start_reduce_epsilon
        self.epoch_num = epoch_num
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sample_obs = self.env.reset()
        self.input_dim = sample_obs.shape # obs is (num_of_days, single_day_features)
        self.action_dim = self.env.action_space
        
        self.Q = self.DuelingBiLSTM(single_day_dim= self.input_dim[1], 
                                    look_back_window= self.input_dim[0], 
                                    action_space_dim= self.action_dim, 
                                    rnn_hidden_space= (self.rnn_hidden_size,), 
                                    fc_hidden_space= (self.fcc_hidden_size,)).to(self.device)
        self.Q_target = copy.deepcopy(self.Q).to(self.device)
        self.optimizer = optim.Adam(self.Q.parameters())
        

        
        self.memory = self.PrioritizedReplayBuffer(mem_max_len, mem_min_len)
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
                # select action
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(self.action_dim)
                else:
                    state_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        q_out = self.Q(state_tensor)
                    action = q_out.argmax(dim=1).item()

                # step environment; unpack info if provided
                step_result = self.env.step(action)
                if len(step_result) == 4:
                    next_obs, reward, done, info = step_result
                else:
                    next_obs, reward, done = step_result
                    info = {}

                # store in memory
                self.memory.add((obs, action, reward, next_obs, done))
        
                
                if self.memory.is_min_len_reached() and self.total_step % self.train_freq == 0:
                    for i in range(self.train_times):
                        transition, indices, is_weights = self.memory.sample(self.batch_size, 0.7, 1-self.epsilon)
                        
                        bs, ba, br, bnext, bd = zip(*transition)
                        
                        # print(bs)
                        bs = torch.tensor(bs, dtype=torch.float32, device=self.device)
                        
        
                        
                        ba = torch.tensor(ba, dtype=torch.long, device=self.device)
                        br = torch.tensor(br, dtype=torch.float32, device=self.device)
                        bnext = np.array(bnext)
                        bnext = torch.tensor(bnext, dtype=torch.float32, device=self.device)
                        bd = torch.tensor(bd, dtype=torch.float32, device=self.device)
                        is_weights = torch.tensor(is_weights, dtype=torch.float32, device=self.device)
                        

                        q_values = self.Q(bs)
                        
                        q_val = q_values.gather(1, ba.unsqueeze(1)).squeeze(1)
        
                        # Double DQN target
                        with torch.no_grad():
                            next_q = self.Q(bnext)
                            next_actions = next_q.argmax(dim=1, keepdim=True)
                            q_target_next = self.Q_target(bnext)
                            q_target_val = q_target_next.gather(1, next_actions).squeeze(1)
                            target = br + self.gamma * q_target_val * (1 - bd)
        
                        errors = (target - q_val).detach().cpu().numpy()
                        
                        loss = ((target-q_val) ** 2) * is_weights
                        loss = loss.mean()
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        total_loss += loss.item()

                        
                        self.memory.update_priorities(indices, errors)
        
                        # Debuggin to check if reward is not normalized somehow
                        # if abs(reward) > 10:
                        #     print(f"env_step: {getattr(self.env, 'current_step', 0)}  reward: {reward}    loss: {loss.item()}")

                # update target network
                if self.total_step % self.update_q_freq == 0:
                    self.Q_target.load_state_dict(self.Q.state_dict())

                # epsilon decay
                if self.total_step > self.start_reduce_epsilon:
                    self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
                
                obs = next_obs
                total_reward += reward
                step += 1
                self.total_step += 1

            # end of episode logging
            self.total_rewards.append(total_reward)
            self.total_losses.append(total_loss)
            self.total_profits.append(getattr(self.env, 'total_profit', 0))
            # episode return == total_reward, profit info in env.profits
            print(f"Epoch {epoch+1:3d} | Epsilon {self.epsilon:.3f} | Steps {self.total_step:5d} | Return {total_reward:.4f} | Loss {total_loss:.4f} | Profit {getattr(self.env, 'total_profit', 0):.4f}")
            if hasattr(self.env, "render_confusion_matrix"):
                self.env.render_confusion_matrix()
            

            # periodic summary every 5 episodes
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
        if hasattr(test_env, "render_confusion_matrix"):
                test_env.render_confusion_matrix()
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

