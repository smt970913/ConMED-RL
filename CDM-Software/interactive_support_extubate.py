import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import pandas as pd

import shap

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import math
import os
import copy
import random
from pathlib import Path
import sys
import datetime
from tqdm import tqdm
from collections import Counter

curr_path = str(Path().absolute())
parent_path = str(Path().absolute().parent)
sys.path.append(parent_path) # add current terminal path to sys.path
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # obtain current time

class FCN_fqe(nn.Module):
    def __init__(self, state_dim, action_dim):

        super(FCN_fqe, self).__init__()
        self.fc1 = nn.Linear(state_dim, action_dim)
        # self.fc2 = nn.Linear(500, action_dim)

    def forward(self, x):

        x = self.fc1(x)
        # x = F.elu(x, alpha = 1.0)
        # x = F.leaky_relu(x, negative_slope = 0.1)
        # x = self.fc2(x)

        return x
    
class FCN_fqi(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(FCN_fqi, self).__init__()
        self.fc1 = nn.Linear(state_dim, action_dim)

    def forward(self, x):
        x = self.fc1(x)
        return x
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, obj_cost, con_cost, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state, action, obj_cost, con_cost, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, obj_cost, con_cost, next_state, done = zip(*batch)
        return state, action, obj_cost, con_cost, next_state, done

    def extract(self):
        batch = self.buffer
        state, action, obj_cost, con_cost, next_state, done = zip(*batch)
        return state, action, obj_cost, con_cost, next_state, done

    def simple_extract(self):
        batch = self.buffer
        state, action, obj_cost, con_cost, next_state, done = zip(*batch)
        return state

    def clear(self):
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)

class StratifiedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer_0 = []
        self.buffer_1 = []

    def push(self, state, action, obj_cost, con_cost, next_state, done):
        transition = (state, action, obj_cost, con_cost, next_state, done)
        if action == 0.0:
            self.buffer_0.append(transition)
        else:
            self.buffer_1.append(transition)

    def sample(self, sample_size, ratio = 0.5):
        n_0_sample = int(sample_size * ratio)
        n_1_sample = sample_size - n_0_sample

        sample_transitions_0 = random.sample(self.buffer_0, n_0_sample)
        sample_transitions_1 = random.sample(self.buffer_1, n_1_sample)

        all_transitions = sample_transitions_0 + sample_transitions_1
        random.shuffle(all_transitions)

        state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch = zip(*all_transitions)
        
        return state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch
    
class FQE:
    def __init__(self, cfg, state_dim, action_dim, eval_agent, eval_target = 'obj'):
        self.device = cfg.device
        self.gamma = cfg.gamma

        if eval_target == 'obj':
            self.lr_fqe = cfg.lr_fqe_obj
        else:
            self.lr_fqe = cfg.lr_fqe_con[eval_target] 

        self.policy_net = FCN_fqe(state_dim, action_dim).to(self.device)
        self.target_net = FCN_fqe(state_dim, action_dim).to(self.device)
        # self.policy_net = torch.compile(self.policy_net)
        # self.target_net = torch.compile(self.target_net)

        # initialize target Q-Estimator with policy Q-Estimator
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        self.optimizer = optim.SGD(self.policy_net.parameters(), lr = self.lr_fqe)
        
        # define loss function
        self.loss = cfg.loss_fqe
        
        # input the evaluation agent
        self.eval_agent = eval_agent

    def update(self, state_batch, action_batch, cost_batch, next_state_batch, done_batch):
        with torch.no_grad():
            # We need to evaluate the parameterized policy
            policy_action_batch = self.eval_agent.rl_policy(next_state_batch)
            next_q_values = self.target_net(next_state_batch).gather(dim = 1, index = policy_action_batch).squeeze(1)
            expected_q_values = cost_batch + self.gamma * next_q_values * (1 - done_batch)
            expected_q_values = expected_q_values.unsqueeze(1)
            
        q_values = self.policy_net(state_batch).gather(dim = 1, index = action_batch)

        loss = self.loss(q_values, expected_q_values)
        self.optimizer.zero_grad(set_to_none = True)
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), clip_value = 1.0)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm = 1.0)
        self.optimizer.step()
        return loss.detach()

    def avg_Q_value_est(self, state_batch):
        with torch.no_grad():
            policy_action_batch = self.eval_agent.rl_policy(state_batch)
            q_values = self.policy_net(state_batch).gather(dim = 1, index = policy_action_batch).squeeze(1)
            q_mean = q_values.mean()
        return q_mean.item()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path + 'FQE_policy_network.pth')
        torch.save(self.target_net.state_dict(), path + 'FQE_target_network.pth')

class FQI:
    def __init__(self, cfg, state_dim, action_dim, omega_1 = 0.0, omega_2 = 0.0):
        self.device = cfg.device
        self.gamma = cfg.gamma
        self.lr = cfg.lr_fqi
        self.omega_1 = omega_1  # L1 regularization weight
        self.omega_2 = omega_2  # L2 regularization weight

        self.policy_net = FCN_fqi(state_dim, action_dim).to(self.device)
        self.target_net = FCN_fqi(state_dim, action_dim).to(self.device)
        # self.policy_net = torch.compile(self.policy_net)
        # self.target_net = torch.compile(self.target_net)

        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        self.optimizer = optim.SGD([
            {'params': self.policy_net.fc1.weight, 'weight_decay': self.omega_2},  # omega_2 (L2)
            {'params': self.policy_net.fc1.bias,   'weight_decay': 0.0}
            ], lr = self.lr)
        # self.optimizer = optim.SGD(self.policy_net.parameters(), lr = self.lr)
        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr = self.lr)

        self.loss = cfg.loss_fqi

    def update(self, lambda_t, state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch, l1_reg = True):
        with torch.no_grad():
            policy_action_batch = self.policy_net(next_state_batch).min(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_state_batch).gather(dim = 1, index = policy_action_batch).squeeze(1)
            expected_q_values = (obj_cost_batch + lambda_t * con_cost_batch) + self.gamma * next_q_values * (1 - done_batch)
            expected_q_values = expected_q_values.unsqueeze(1)
            
        q_values = self.policy_net(state_batch).gather(dim = 1, index = action_batch)
        
        loss = self.loss(q_values, expected_q_values)
        self.optimizer.zero_grad(set_to_none = True)
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), clip_value = 1.0)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm = 1.0)
        self.optimizer.step()
        
        if l1_reg:
            # Proximal (L1) step: Apply soft-thresholding for L1 regularization
            if self.omega_1 > 0:
                with torch.no_grad():
                    threshold = self.lr * self.omega_1
                    # Apply soft-thresholding only to weights, not bias
                    weight = self.policy_net.fc1.weight
                    weight.data = torch.sign(weight.data) * torch.clamp(torch.abs(weight.data) - threshold, min = 0.0)

        return loss.detach()

    def get_sparsity(self):
        with torch.no_grad():
            weight = self.policy_net.fc1.weight
            total_params = weight.numel()
            zero_params = (weight.abs() < 1e-6).sum().item()
            sparsity = zero_params / total_params
        return sparsity

    def avg_Q_value_est(self, state_batch):
        with torch.no_grad():
            q_values = self.policy_net(state_batch)
            avg_q_values = q_values.min(1)[0].unsqueeze(1).mean().item()
        return avg_q_values

    def rl_policy(self, state_batch):
        with torch.no_grad():
            q_values = self.policy_net(state_batch)
            policy_action_batch = q_values.min(1)[1].unsqueeze(1)
        return policy_action_batch

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path + 'Offline_FQI_policy_network.pth')
        torch.save(self.target_net.state_dict(), path + 'Offline_FQI_target_network.pth')

class RLConfig:
    def __init__(self, algo_name, train_eps, gamma, 
                 lr_fqi, lr_fqe_obj, constraint_num, lr_fqe_con_list, lr_lambda_list, 
                 threshold_list, sample_col_name, sample_method, sample_size, sample_ratio):
        
        self.algo = algo_name  # name of algorithm
        self.train_eps = train_eps  #the number of trainng episodes
        self.gamma = gamma # discount factor
        self.constraint_num = constraint_num

        # learning rates
        self.lr_fqi = lr_fqi
        self.lr_fqe_obj = lr_fqe_obj
        self.lr_fqe_con = [0 for i in range(constraint_num)]
        self.lr_lam = [0 for i in range(constraint_num)]

        # constraint threshold
        self.constraint_limit = [0 for i in range(constraint_num)]
        for i in range(constraint_num):
            self.lr_fqe_con[i] = lr_fqe_con_list[i]
            self.lr_lam[i] = lr_lambda_list[i]
            self.constraint_limit[i] = threshold_list[i]

        self.train_eps_steps = int(1e3)  # the number of steps in each training episode

        self.batch_size = sample_size
        
        self.sample_col_name = sample_col_name
        self.sample_method = sample_method
        self.sample_size = int(sample_size * 0.2)
        self.sample_ratio = sample_ratio

        self.loss_fqi = nn.MSELoss()
        self.loss_fqe = nn.MSELoss()

        self.memory_capacity = int(5e6)  # capacity of Replay Memory

        self.target_update = 100 # update frequency of target net
        self.tau = 0.01

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check GPU
        self.device = torch.device("cpu")

class DataLoader:
    def __init__(self, cfg, state_id_table, rl_cont_state_table, rl_cont_state_table_scaled, terminal_state):
        self.cfg = cfg
    
        # Load datasets
        self.state_df_id = state_id_table
        self.rl_cont_state_table = rl_cont_state_table
        self.rl_cont_state_table_scaled = rl_cont_state_table_scaled

        self.terminal_state = terminal_state

        # Add new properties for bootstrap
        self.unique_episodes = []
        self.M_train_episodes = 0
        self.episode_to_transition_indices = {}

    def data_buffer_train(self):
        self.train_memory_normal = ReplayBuffer(self.cfg.memory_capacity)
        self.train_memory_stratify = StratifiedReplayBuffer(self.cfg.memory_capacity)

        for i in range(len(self.state_df_id)):
            state = self.rl_cont_state_table_scaled.values[i]
            action = self.state_df_id['ext_action'].values[i]
            
            if action == 1.0:
                done = 1.0
            else:
                done = 0.0
            
            obj_cost = self.state_df_id['extubation_fail_costs'].values[i]
            con_cost = self.state_df_id['con_cost_0'].values[i]

            if done == 0.0:
                idx = self.state_df_id.index[i]
                next_state = self.rl_cont_state_table_scaled.loc[idx + 1].values
            else:
                next_state = self.terminal_state

            self.train_memory_normal.push(state, action, obj_cost, con_cost, next_state, done)
            self.train_memory_stratify.push(state, action, obj_cost, con_cost, next_state, done)

    def data_torch_loader_train(self, bootstrap_data_index, bootstrap_method = True, sample_method = 'random', ratio = 0.5):
        if bootstrap_method:
            if sample_method == 'random':
                state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch = self.bootstrap_buffers[bootstrap_data_index].sample(self.cfg.sample_size)
            else:
                state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch = self.bootstrap_buffers[bootstrap_data_index].sample(self.cfg.sample_size, ratio)
        else:
            if sample_method == 'random':
                state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch = self.train_memory_normal.sample(self.cfg.batch_size)
            else:
                state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch = self.train_memory_stratify.sample(self.cfg.batch_size, ratio)

        state_batch = torch.tensor(np.array(state_batch), device = self.cfg.device, dtype = torch.float)
        action_batch = torch.tensor(np.array(action_batch), device = self.cfg.device, dtype = torch.long).unsqueeze(1)
        obj_cost_batch = torch.tensor(np.array(obj_cost_batch), device = self.cfg.device, dtype = torch.float)
        con_cost_batch = torch.tensor(np.array(con_cost_batch), device = self.cfg.device, dtype = torch.float)
        next_state_batch = torch.tensor(np.array(next_state_batch), device = self.cfg.device, dtype = torch.float)
        done_batch = torch.tensor(np.array(done_batch), device = self.cfg.device, dtype = torch.float)

        return state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch

    def prepare_bootstrap(self, episode_col_name):
        if episode_col_name not in self.state_df_id.columns:
            raise ValueError(f"Episode column '{episode_col_name}' not found in state_df_id.")
        self.episode_to_transition_indices = self.state_df_id.groupby(episode_col_name).apply(lambda x: x.index.tolist()).to_dict()
        self.unique_episodes = list(self.episode_to_transition_indices.keys())
        self.M_train_episodes = len(self.unique_episodes)
        print(f"Bootstrap prepared: Found {self.M_train_episodes} unique episodes.")

    def create_bootstrap_dataset_indices(self):
        if not self.unique_episodes:
            raise RuntimeError("prepare_bootstrap() must be called before creating dataset indices.")
        # Sample M_train episodes with replacement (core part)
        sampled_episode_ids = np.random.choice(
            self.unique_episodes,
            size = self.M_train_episodes,
            replace = True
        )
        # Get all transition indices from these episodes
        bootstrap_transition_indices = []
        for ep_id in sampled_episode_ids:
            bootstrap_transition_indices.extend(self.episode_to_transition_indices[ep_id])
        return bootstrap_transition_indices

    def construct_bootstrap_buffer(self, number_of_bootstrap):
        self.bootstrap_buffers = {i: [] for i in range(number_of_bootstrap)}
        for i in range(number_of_bootstrap):
            bootstrap_buffer = ReplayBuffer(self.cfg.memory_capacity)
            bootstrap_indices = self.create_bootstrap_dataset_indices()
            for idx in bootstrap_indices:
                bootstrap_buffer.push(*self.train_memory_normal.buffer[idx])
            self.bootstrap_buffers[i] = bootstrap_buffer

class ValDataLoader:
    def __init__(self, cfg, state_id_table, rl_cont_state_table, rl_cont_state_table_scaled, 
                 state_id_table_1, rl_cont_state_table_scaled_1, terminal_state):
        self.cfg = cfg
    
        # Load datasets
        self.state_df_id = state_id_table
        self.rl_cont_state_table = rl_cont_state_table
        self.rl_cont_state_table_scaled = rl_cont_state_table_scaled

        self.state_df_id_1 = state_id_table_1
        self.rl_cont_state_table_scaled_1 = rl_cont_state_table_scaled_1

        self.terminal_state = terminal_state

    def data_buffer_val(self):
        self.val_memory = ReplayBuffer(self.cfg.memory_capacity)

        for i in range(len(self.state_df_id)):
            state = self.rl_cont_state_table_scaled.values[i]
            action = self.state_df_id['ext_action'].values[i]
            
            if action == 1.0:
                done = 1.0
            else:
                done = 0.0
            
            obj_cost = self.state_df_id['extubation_fail_costs'].values[i]
            con_cost = self.state_df_id['con_cost_0'].values[i]

            if done == 0.0:
                idx = self.state_df_id.index[i]
                next_state = self.rl_cont_state_table_scaled_1.loc[idx + 1].values
            else:
                next_state = self.terminal_state

            self.val_memory.push(state, action, obj_cost, con_cost, next_state, done)

    def data_torch_loader_val(self):
        state_batch = self.val_memory.simple_extract()
        state_batch = torch.tensor(np.array(state_batch), device = self.cfg.device, dtype = torch.float)
        return state_batch

class RLTraining:
    def __init__(self, cfg, state_dim, action_dim, train_data_loader, val_data_loader, ensemble_size = 10):
        self.cfg = cfg
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader

        # Ensemble sizes for bootstrap FQE
        self.ensemble_size_obj = ensemble_size
        self.ensemble_size_con_list = [ensemble_size] * cfg.constraint_num

    def fqi_agent_config(self, omega_1, omega_2, seed = 1):
        agent_fqi = FQI(self.cfg, self.state_dim, self.action_dim, omega_1, omega_2)
        torch.manual_seed(seed)
        return agent_fqi

    def fqe_agent_config(self, eval_agent, eval_target, seed = 1):
        agent_fqe = FQE(self.cfg, self.state_dim, self.action_dim, eval_agent, eval_target)
        torch.manual_seed(seed)
        return agent_fqe
    
    # Create ensemble of FQE agents for both objective and constraint costs
    def fqe_ensemble_config(self, eval_agent, eval_target, base_seed = 1):
        ensemble = []
        ensemble_size = 1
        
        if eval_target == 'obj':
            ensemble_size = self.ensemble_size_obj
        else:
            ensemble_size = self.ensemble_size_con_list[eval_target]
            
        for e in range(ensemble_size):
            fqe_seed = base_seed + e * 1000  # Different seed for each ensemble member
            agent = self.fqe_agent_config(eval_agent, eval_target, fqe_seed)
            ensemble.append(agent)
        return ensemble
    
    def prepare_bootstrap_data_buffer(self):
        self.train_data_loader.construct_bootstrap_buffer(self.ensemble_size_obj)
        
    def compute_bootstrap_stats(self, ests_list, confidence_level = 0.98):
        if len(ests_list) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [0.0, 0.0]
        
        ests_array = np.array(ests_list)
        E = len(ests_list)
        
        # Point estimate (mean)
        mean = np.mean(ests_array)
        
        # Sample standard deviation
        if E > 1:
            std = np.std(ests_array, ddof = 1)  # sample std (ddof=1 for unbiased estimate)
        else:
            std = 0.0
        
        # Standard error
        se = std / np.sqrt(E) if E > 0 else 0.0
        
        # Z-value for confidence interval
        alpha = 1 - confidence_level
        z = 2.33  # For 98% CI, or use scipy.stats.norm.ppf(1 - alpha/2)
        if confidence_level == 0.95:
            z = 1.96
        elif confidence_level == 0.99:
            z = 2.58
        
        # Upper confidence bound
        upper_bound = mean + z * se
        
        # Normal approximation CI
        ci_lower = mean - z * se
        ci_upper = mean + z * se
        
        # Additional: Percentile bootstrap CI
        sorted_ests = np.sort(ests_array)
        lower_idx = int(np.ceil(E * alpha / 2)) - 1
        upper_idx = int(np.ceil(E * (1 - alpha / 2))) - 1
        lower_idx = max(0, lower_idx)
        upper_idx = min(E - 1, upper_idx)
        percentile_ci = [sorted_ests[lower_idx], sorted_ests[upper_idx]]
        
        return mean, std, se, upper_bound, ci_lower, ci_upper, percentile_ci

    def compute_bootstrap_stats_simple(self, ests_list, confidence_level = 0.98):
        if len(ests_list) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [0.0, 0.0]
        
        ests_array = np.array(ests_list)
        E = len(ests_list)
        
        # Point estimate (mean)
        mean = np.mean(ests_array)
        
        # Sample standard deviation
        if E > 1:
            std = np.std(ests_array, ddof = 1)  # sample std (ddof=1 for unbiased estimate)
        else:
            std = 0.0
        
        # Standard error
        se = std / np.sqrt(E) if E > 0 else 0.0
        
        # Z-value for confidence interval
        alpha = 1 - confidence_level
        z = 2.33  # For 98% CI, or use scipy.stats.norm.ppf(1 - alpha/2)
        if confidence_level == 0.95:
            z = 1.96
        elif confidence_level == 0.99:
            z = 2.58
        
        # Upper confidence bound
        upper_bound = mean + z * se
        
        return mean, upper_bound

    def train(self, agent_fqi, agent_fqe_obj, agent_fqe_con, constraint = True, fqi_sample = 'stratify', l1_reg = True):
        print('Start to train!')
        print(f'Algorithm:{self.cfg.algo}, Device:{self.cfg.device}')
        print(f'Bootstrap ensemble enabled: Obj E = {self.ensemble_size_obj}, Con E = {self.ensemble_size_con_list}')

        self.FQI_loss = []
        self.FQE_loss_obj = []
        self.FQE_loss_con = []
        
        self.FQI_est_values = []
        self.FQE_est_obj_costs = []
        self.FQE_est_obj_costs_stats = []  # Store bootstrap statistics

        self.FQE_est_con_costs = []
        self.FQE_est_con_costs_stats = []

        self.lambda_dict = []

        lambda_t = 0
        lambda_update = 0

        state_batch_val = self.val_data_loader.data_torch_loader_val()

        for k in range(self.cfg.train_eps):
            list_fqe_obj_values = []
            list_fqe_con_values = []
            for j in tqdm(range(self.cfg.train_eps_steps), desc = f"Epoch {k + 1}/{self.cfg.train_eps}"):
                state_batch, action_batch, \
                    obj_cost_batch, con_cost_batch, next_state_batch, done_batch = self.train_data_loader.data_torch_loader_train(
                    bootstrap_data_index = 0, bootstrap_method = False, sample_method = fqi_sample, ratio = 0.5
                    )
                loss_rl = agent_fqi.update(
                    lambda_t, state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch, l1_reg
                    )
                loss_obj_e = agent_fqe_obj.update(
                    state_batch, action_batch, obj_cost_batch, next_state_batch, done_batch
                )
                loss_con_e = agent_fqe_con.update(
                    state_batch, action_batch, con_cost_batch, next_state_batch, done_batch
                )

                obj_est_value = agent_fqe_obj.avg_Q_value_est(state_batch_val)
                con_est_value = agent_fqe_con.avg_Q_value_est(state_batch_val)

                list_fqe_obj_values.append(obj_est_value)
                list_fqe_con_values.append(con_est_value)

                if constraint == False:                        
                    lambda_update = 0
                    lambda_t = 0
                else:
                    # lambda_update = upper_bound_con - self.cfg.constraint_limit[0]
                    lambda_update = con_est_value - self.cfg.constraint_limit[0]
                    lambda_t = lambda_t + (self.cfg.lr_lam[0] * lambda_update)
                    lambda_t = max(0, lambda_t)

                if (j + 1) % self.cfg.target_update == 0:                    
                    self._soft_update(agent_fqi.policy_net, agent_fqi.target_net, self.cfg.tau)
                    self._soft_update(agent_fqe_obj.policy_net, agent_fqe_obj.target_net, self.cfg.tau)
                    self._soft_update(agent_fqe_con.policy_net, agent_fqe_con.target_net, self.cfg.tau)

            self.FQI_loss.append(loss_rl.item())
            self.FQI_est_values.append(agent_fqi.avg_Q_value_est(state_batch_val))
           
            self.FQE_loss_obj.append(loss_obj_e.item())
            # self.FQE_est_obj_costs.append(obj_est_value)
            self.FQE_est_obj_costs.append(np.mean(list_fqe_obj_values))

            self.FQE_loss_con.append(loss_con_e.item())
            # self.FQE_est_con_costs.append(con_est_value)
            self.FQE_est_con_costs.append(np.mean(list_fqe_con_values))

            self.lambda_dict.append(lambda_t)
            print(f"Average FQE estimated constraint cost of objective-EFR after epoch {k + 1}: {np.mean(list_fqe_obj_values):.4f}")
            print(f"Average FQE estimated constraint cost of constraint-ICU LOS after epoch {k + 1}: {np.mean(list_fqe_con_values):.4f}")
            print(f"Dual variable after epoch {k + 1}: {lambda_t}")
            print(f"Dual variable update after epoch {k + 1}: {lambda_update}")

        print("Complete Training!")
    
    def _soft_update(self, policy_net, target_net, tau):
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)
    
    def _get_model_state(self, agent):
        """
        Extract the model state from an agent.
        Returns a deep copy of both policy_net and target_net states.
        """
        return {
            'policy_net': copy.deepcopy(agent.policy_net.state_dict()),
            'target_net': copy.deepcopy(agent.target_net.state_dict())
        }
    
    def _save_models_to_disk(self):
        """
        Save the models to disk.
        This method can be customized based on your specific requirements.
        """       
        # Create directory for saved models if it doesn't exist
        os.makedirs('saved_models/fqe_obj', exist_ok = True)
        
        # Save objective FQE models
        for idx, model_data in enumerate(self.fqe_obj_models_history[-2000:]):
            torch.save(
                model_data['model_state'], 
                f'saved_models/fqe_obj/model_{model_data["update_num"]}.pt'
            )
        
        # Save constraint FQE models
        for con_idx in self.fqe_con_models_history.keys():
            os.makedirs(f'saved_models/fqe_con_{con_idx}', exist_ok = True)
            
            for idx, model_data in enumerate(self.fqe_con_models_history[con_idx][-2000:]):
                torch.save(
                    model_data['model_state'], 
                    f'saved_models/fqe_con_{con_idx}/model_{model_data["update_num"]}.pt'
                )
        
        print("Models saved to disk in 'saved_models/' directory")
    
    def load_fqe_model(self, agent, model_path):
        """
        Load a saved FQE model into an agent.
        
        Args:
            agent: The FQE agent to load the model into
            model_path: Path to the saved model state
        
        Returns:
            The agent with loaded model
        """
        model_state = torch.load(model_path)
        agent.policy_net.load_state_dict(model_state['policy_net'])
        agent.target_net.load_state_dict(model_state['target_net'])
        return agent

class TestDataLoader:
    def __init__(self, cfg, state_id_table, rl_cont_state_table, rl_cont_state_table_scaled, 
                 state_id_table_1, rl_cont_state_table_scaled_1, terminal_state):
        self.cfg = cfg
        
        # Load datasets
        self.state_df_id = state_id_table
        self.rl_cont_state_table = rl_cont_state_table
        self.rl_cont_state_table_scaled = rl_cont_state_table_scaled

        self.state_df_id_1 = state_id_table_1
        self.rl_cont_state_table_scaled_1 = rl_cont_state_table_scaled_1

        self.terminal_state = terminal_state

    def data_buffer_test(self, num_constraint = 1):
        self.test_memory = ReplayBuffer(self.cfg.memory_capacity)

        for i in range(len(self.state_df_id)):
            state = self.rl_cont_state_table_scaled.values[i]
            action = self.state_df_id['ext_action'].values[i]
            
            if action == 1.0:
                done = 1.0
            else:
                done = 0.0
            
            obj_cost = self.state_df_id['extubation_fail_costs'].values[i]
            con_cost = self.state_df_id['con_cost_0'].values[i]

            if done == 0.0:
                idx = self.state_df_id.index[i]
                next_state = self.rl_cont_state_table_scaled_1.loc[idx + 1].values
            else:
                next_state = self.terminal_state

            self.test_memory.push(state, action, obj_cost, con_cost, next_state, done)

    def data_torch_loader_test(self):
        state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch = self.test_memory.extract()

        state_batch = torch.tensor(np.array(state_batch), device = self.cfg.device, dtype = torch.float)
        action_batch = torch.tensor(np.array(action_batch), device = self.cfg.device, dtype = torch.long).unsqueeze(1)
        obj_cost_batch = torch.tensor(np.array(obj_cost_batch), device = self.cfg.device, dtype = torch.float)
        con_cost_batch = torch.tensor(np.array(con_cost_batch), device = self.cfg.device, dtype = torch.float)
        next_state_batch = torch.tensor(np.array(next_state_batch), device = self.cfg.device, dtype = torch.float)
        done_batch = torch.tensor(np.array(done_batch), device = self.cfg.device, dtype = torch.float)

        return state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch

class TestConfig:
    def __init__(self, constraint_num):
        
        self.constraint_num = constraint_num

        self.memory_capacity = int(3e6)  # capacity of Replay Memory

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check GPU
        # self.device = torch.device("cpu")