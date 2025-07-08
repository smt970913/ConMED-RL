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
        self.fc1 = nn.Linear(state_dim, 500)
        self.fc2 = nn.Linear(500, action_dim)

    def forward(self, x):

        x = self.fc1(x)
        # x = F.elu(x, alpha = 1.0)
        x = F.leaky_relu(x, negative_slope = 0.1)
        x = self.fc2(x)

        return x
    
class FCN_fqi(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(FCN_fqi, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope = 0.1)
        x = self.fc2(x)
        x = F.leaky_relu(x, negative_slope = 0.1)
        x = self.fc3(x)
        x = F.leaky_relu(x, negative_slope = 0.1)
        x = self.fc4(x)
        return x
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, obj_cost, con_cost, next_state, done):

        if not isinstance(con_cost, list) and not isinstance(con_cost, tuple):
            con_cost = [con_cost]

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state, action, obj_cost, con_cost, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, obj_cost, con_cost, next_state, done = zip(*batch)

        con_cost = [list(costs) for costs in zip(*con_cost)]

        return state, action, obj_cost, con_cost, next_state, done

    def extract(self):
        batch = self.buffer
        state, action, obj_cost, con_cost, next_state, done = zip(*batch)

        con_cost = [list(costs) for costs in zip(*con_cost)]

        return state, action, obj_cost, con_cost, next_state, done

    def clear(self):
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)
    
class FQE:
    def __init__(self, cfg, state_dim, action_dim, id_stop, eval_agent, weight_decay, eval_target = 'obj'):

        self.device = cfg.device

        self.gamma = cfg.gamma

        ### indicate optimal stopping structure or not
        self.id_stop = id_stop

        ### For constraint cost, specify which constraint to evaluate
        if eval_target == 'obj':
            self.lr_fqe = cfg.lr_fqe_obj
        else:
            self.lr_fqe = cfg.lr_fqe_con[eval_target] 

        # define policy Q-Estimator
        self.policy_net = FCN_fqe(state_dim, action_dim).to(self.device)
        # define target Q-Estimator
        self.target_net = FCN_fqe(state_dim, action_dim).to(self.device)

        # initialize target Q-Estimator with policy Q-Estimator
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        self.weight_decay = weight_decay
        # self.optimizer = optim.SGD(self.policy_net.parameters(), lr = self.lr_fqe)
        # self.optimizer = optim.SGD([
        #     {'params': self.policy_net.fc1.weight, 'weight_decay': self.weight_decay}, 
        #     {'params': self.policy_net.fc1.bias,   'weight_decay': 0.0}
        #     ], lr = self.lr_fqe)
        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr = self.lr_fqe)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), 
                                     lr = self.lr_fqe, 
                                     weight_decay = self.weight_decay)

        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max = 100)
        
        # define loss function
        self.loss = cfg.loss_fqe
        
        # input the evaluation agent
        self.eval_agent = eval_agent

    def update(self, state_batch, action_batch, cost_batch, next_state_batch, done_batch, disch_batch):

        # We need to evaluate the parameterized policy
        policy_action_batch = self.eval_agent.rl_policy(next_state_batch)

        # predicted Q-value using policy Q-network
        q_values = self.policy_net(state_batch).gather(dim = 1, index = action_batch)

        # target Q-value calculated by target Q-network
        next_q_values = self.target_net(next_state_batch).gather(dim = 1, index = policy_action_batch).squeeze(1).detach()
        
        if self.id_stop == 0:
            expected_q_values = cost_batch + self.gamma * next_q_values * (1 - done_batch)
        else:
            expected_q_values = cost_batch + self.gamma * next_q_values * (1 - disch_batch)
            
        loss = self.loss(q_values, expected_q_values.unsqueeze(1))

        # Update reward Q-network by minimizing the above loss function
        self.optimizer.zero_grad()
        loss.backward()
        
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        return loss.item()

    def avg_Q_value_est(self, state_batch):
        # policy_action_batch = self.eval_agent.rl_policy(state_batch)
        # q_values = self.policy_net(state_batch).gather(dim = 1, index = policy_action_batch).squeeze(1)

        # q_mean = q_values.mean()
        # q_std = q_values.std()
        # n = q_values.shape[0]
    
        # if n <= 1 or q_std == 0:
        #     return q_mean.item(), q_mean.item()
    
        # z = 2.33  
        # q_upper_bound = q_mean + z * (q_std / math.sqrt(n))
    
        # return q_mean.item(), q_upper_bound.item()
        q_values = self.policy_net(state_batch)  
    
        
        q_mean_per_action = q_values.mean(dim = 0)  
        # q_std_per_action = q_values.std(dim = 0)    
        # n = q_values.shape[0]  # batch_size
        
        # if n <= 1:
        #     return q_mean_per_action, q_mean_per_action
        
        # z = 2.33
        
        # q_upper_bound_per_action = q_mean_per_action + z * (q_std_per_action / math.sqrt(n))
        
        
        # zero_std_mask = (q_std_per_action == 0)
        # q_upper_bound_per_action[zero_std_mask] = q_mean_per_action[zero_std_mask]
        
        return q_mean_per_action[1].item()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path + 'FQE_policy_network.pth')
        torch.save(self.target_net.state_dict(), path + 'FQE_target_network.pth')

class FQI:
    def __init__(self, cfg, state_dim, action_dim):
        
        self.device = cfg.device
        self.gamma = cfg.gamma
        
        self.lr = cfg.lr_fqi

        self.policy_net = FCN_fqi(state_dim, action_dim).to(self.device)
        self.target_net = FCN_fqi(state_dim, action_dim).to(self.device)

        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        # self.optimizer = optim.SGD(self.policy_net.parameters(), lr = self.lr)
        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr = self.lr)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr = self.lr, weight_decay = 1e-2)

        self.loss = cfg.loss_fqi

    def update(self, lambda_t_list, state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch, disch_batch):

        q_values = self.policy_net(state_batch).gather(dim = 1, index = action_batch)
        policy_action_batch = self.policy_net(next_state_batch).min(1)[1].unsqueeze(1)
        next_q_values = self.target_net(next_state_batch).gather(dim = 1, index = policy_action_batch).squeeze(1).detach()

        sum_con_cost = 0
        for i in range(len(lambda_t_list)):
            lambda_t = lambda_t_list[i]
            sum_con_cost += lambda_t * con_cost_batch[i]

        expected_q_values = (obj_cost_batch + sum_con_cost) + self.gamma * next_q_values * (1 - done_batch)

        loss = self.loss(q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def avg_Q_value_est(self, state_batch):

        q_values = self.policy_net(state_batch)
        avg_q_values = q_values.min(1)[0].unsqueeze(1).mean().item()

        return avg_q_values

    def rl_policy(self, state_batch):

        q_values = self.policy_net(state_batch)
        policy_action_batch = q_values.min(1)[1].unsqueeze(1)

        return policy_action_batch

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path + 'Offline_FQI_policy_network.pth')
        torch.save(self.target_net.state_dict(), path + 'Offline_FQI_target_network.pth')

class RLConfig:
    def __init__(self, algo_name, train_eps, gamma, lr_fqi, lr_fqe_obj, constraint_num, lr_fqe_con_list, lr_lambda_list, threshold_list):
        
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

        self.batch_size = 256

        self.loss_fqi = nn.MSELoss()
        self.loss_fqe = nn.MSELoss()

        self.memory_capacity = int(2e6)  # capacity of Replay Memory

        self.target_update = 100 # update frequency of target net
        self.tau = 0.01

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check GPU
        # self.device = torch.device("cpu")

class DataLoader:
    def __init__(self, cfg, state_id_table, rl_cont_state_table, rl_cont_state_table_scaled, terminal_state):
        self.cfg = cfg
    
        # Load datasets
        self.state_df_id = state_id_table
        self.rl_cont_state_table = rl_cont_state_table
        self.rl_cont_state_table_scaled = rl_cont_state_table_scaled

        self.terminal_state = terminal_state

    def data_buffer_train(self, num_constraint = 2):
        self.train_memory = ReplayBuffer(self.cfg.memory_capacity)

        for i in range(len(self.state_df_id)):
            state = self.rl_cont_state_table_scaled.values[i]
            action = self.state_df_id['discharge_action'].values[i]
            
            if action == 1.0:
                if self.state_df_id['death'].values[i] == 1.0:
                    if self.state_df_id['discharge_fail'].values[i] == 1.0:
                        done = 0.0
                    else:
                        done = 1.0
                else:
                    if self.state_df_id['discharge_fail'].values[i] == 0.0:
                        done = 1.0
                    else:
                        done = 0.0
            else:
                done = 0.0
            
            obj_cost = self.state_df_id['mortality_costs_md'].values[i]
            con_cost = []
            
            for j in range(num_constraint):
                cost_col = f'con_cost_{j}'  
                if cost_col in self.state_df_id.columns:
                    con_cost.append(self.state_df_id[cost_col].values[i])
                else:
                    con_cost.append(0.0) 

            if done == 0.0:
                idx = self.state_df_id.index[i]
                next_state = self.rl_cont_state_table_scaled.loc[idx + 1].values
            else:
                next_state = self.terminal_state

            self.train_memory.push(state, action, obj_cost, con_cost, next_state, done)

    def data_torch_loader_train(self):
        state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch = self.train_memory.sample(self.cfg.batch_size)

        state_batch = torch.tensor(np.array(state_batch), device = self.cfg.device, dtype = torch.float)

        disch_batch = list(action_batch)
        action_batch = torch.tensor(np.array(action_batch), device = self.cfg.device, dtype = torch.long).unsqueeze(1)
        
        obj_cost_batch = torch.tensor(np.array(obj_cost_batch), device = self.cfg.device, dtype = torch.float)
        con_cost_batch = [torch.tensor(np.array(cost), device = self.cfg.device, dtype=torch.float) for cost in con_cost_batch]
        next_state_batch = torch.tensor(np.array(next_state_batch), device = self.cfg.device, dtype = torch.float)
        
        done_batch = torch.tensor(np.array(done_batch), device = self.cfg.device, dtype = torch.float)
        disch_batch = torch.tensor(np.array(disch_batch), device = self.cfg.device, dtype = torch.float)

        return state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch, disch_batch
    
class ValDataLoader:
    def __init__(self, cfg, state_id_table, rl_cont_state_table, rl_cont_state_table_scaled, state_id_table_1, rl_cont_state_table_scaled_1, terminal_state):
        self.cfg = cfg
    
        # Load datasets
        self.state_df_id = state_id_table
        self.rl_cont_state_table = rl_cont_state_table
        self.rl_cont_state_table_scaled = rl_cont_state_table_scaled

        self.state_df_id_1 = state_id_table_1
        self.rl_cont_state_table_scaled_1 = rl_cont_state_table_scaled_1

        self.terminal_state = terminal_state

    def data_buffer_val(self, num_constraint = 2):
        self.val_memory = ReplayBuffer(self.cfg.memory_capacity)

        for i in range(len(self.state_df_id)):
            state = self.rl_cont_state_table_scaled.values[i]
            action = self.state_df_id['discharge_action'].values[i]
            
            if action == 1.0:
                if self.state_df_id['death'].values[i] == 1.0:
                    if self.state_df_id['discharge_fail'].values[i] == 1.0:
                        done = 0.0
                    else:
                        done = 1.0
                else:
                    if self.state_df_id['discharge_fail'].values[i] == 0.0:
                        done = 1.0
                    else:
                        done = 0.0
            else:
                done = 0.0
            
            obj_cost = self.state_df_id['mortality_costs_md'].values[i]
            con_cost = []
            
            for j in range(num_constraint):
                cost_col = f'con_cost_{j}'  
                if cost_col in self.state_df_id.columns:
                    con_cost.append(self.state_df_id[cost_col].values[i])
                else:
                    con_cost.append(0.0) 

            if done == 0.0:
                idx = self.state_df_id.index[i]
                next_state = self.rl_cont_state_table_scaled_1.loc[idx + 1].values
            else:
                next_state = self.terminal_state

            self.val_memory.push(state, action, obj_cost, con_cost, next_state, done)

    def data_torch_loader_val(self):
        state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch = self.val_memory.extract()

        state_batch = torch.tensor(np.array(state_batch), device = self.cfg.device, dtype = torch.float)

        return state_batch
    
class RLTraining:
    def __init__(self, cfg, state_dim, action_dim, data_loader, val_data_loader):
        self.cfg = cfg
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.data_loader = data_loader
        self.val_data_loader = val_data_loader

        # Store for FQI models history
        self.fqi_models_history = []
        
        # Store for FQE models history
        self.fqe_obj_models_history = []
        self.fqe_con_models_history = {}

    def fqi_agent_config(self, seed = 1):
        agent_fqi = FQI(self.cfg, self.state_dim, self.action_dim)
        torch.manual_seed(seed)
        return agent_fqi

    def fqe_agent_config(self, id_stop, eval_agent, weight_decay, eval_target, seed = 2):
        agent_fqe = FQE(self.cfg, self.state_dim, self.action_dim, id_stop, eval_agent, weight_decay, eval_target)
        torch.manual_seed(seed)
        return agent_fqe

    def train(self, agent_fqi, agent_fqe_obj, agent_fqe_con_list, constraint = None):
        print('Start to train!')
        print(f'Algorithm:{self.cfg.algo}, Device:{self.cfg.device}')

        self.FQI_loss = []
        self.FQI_est_values = []

        self.FQE_loss_obj = []
        self.FQE_loss_con = {i: [] for i in range(len(agent_fqe_con_list))}

        self.FQE_est_obj_costs = []
        self.FQE_est_con_costs = {i: [] for i in range(len(agent_fqe_con_list))}

        self.lambda_dict = {i: [] for i in range(len(agent_fqe_con_list))}
        
        # Initialize the model history dictionaries for constraint agents
        for i in range(len(agent_fqe_con_list)):
            self.fqe_con_models_history[i] = []

        lambda_t_list = [0 for i in range(len(agent_fqe_con_list))]
        lambda_update_list = [0 for i in range(len(agent_fqe_con_list))]

        state_batch_val = self.val_data_loader()
        
        model_update_counter = 0  # Counter to track model updates

        for k in range(self.cfg.train_eps):
            loss_list_fqi = []
            loss_list_fqe_obj = []
            loss_list_fqe_con = {i: [] for i in range(len(agent_fqe_con_list))}

            fqi_est_list = []
            fqe_est_obj = []
            fqe_est_con = {i: [] for i in range(len(agent_fqe_con_list))}

            for j in tqdm(range(self.cfg.train_eps_steps)):

                state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch, disch_batch = self.data_loader()
                
                # update the policy agent for learning agent (FQI) and evaluation agent (FQE)
                loss_rl = agent_fqi.update(lambda_t_list, state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch, disch_batch)
                loss_ev_obj = agent_fqe_obj.update(state_batch, action_batch, obj_cost_batch, next_state_batch, done_batch, disch_batch)
                
                # Save FQI model state
                model_update_counter += 1
                if len(self.fqi_models_history) >= 100:
                    self.fqi_models_history.pop(0)  
                self.fqi_models_history.append({
                    'update_num': model_update_counter,
                    'epoch': k,
                    'step': j,
                    'model_state': self._get_model_state(agent_fqi)
                })

                # Save FQE objective model state
                if len(self.fqe_obj_models_history) >= 100:
                    self.fqe_obj_models_history.pop(0) 
                self.fqe_obj_models_history.append({
                    'update_num': model_update_counter,
                    'epoch': k,
                    'step': j,
                    'model_state': self._get_model_state(agent_fqe_obj)
                })
                
                ##############################################################################################################
                loss_list_fqi.append(loss_rl)
                loss_list_fqe_obj.append(loss_ev_obj)

                if constraint == None:
                    for m in range(len(agent_fqe_con_list)):
                        loss_con = agent_fqe_con_list[m].update(state_batch, action_batch, con_cost_batch[m], next_state_batch, done_batch, disch_batch)
                        loss_list_fqe_con[m].append(loss_con)
                        
                        # Save FQE constraint model state
                        if len(self.fqe_con_models_history[m]) >= 100:
                            self.fqe_con_models_history[m].pop(0)  
                        self.fqe_con_models_history[m].append({
                            'update_num': model_update_counter,
                            'epoch': k,
                            'step': j,
                            'model_state': self._get_model_state(agent_fqe_con_list[m])
                        })
                        
                        con_est_value, con_est_value_up = agent_fqe_con_list[m].avg_Q_value_est(state_batch_val)
                        fqe_est_con[m].append(con_est_value)
                    
                    fqi_est_value = agent_fqi.avg_Q_value_est(state_batch_val)
                    avg_q_value_obj, avg_q_value_obj_up = agent_fqe_obj.avg_Q_value_est(state_batch_val)

                    fqi_est_list.append(fqi_est_value)
                    fqe_est_obj.append(avg_q_value_obj)

                    lambda_update_list = [0 for i in range(len(agent_fqe_con_list))]
                    lambda_t_list = [0 for i in range(len(agent_fqe_con_list))]
                
                else:
                    for m in range(len(agent_fqe_con_list)):
                        loss_con = agent_fqe_con_list[m].update(state_batch, action_batch, con_cost_batch[m], next_state_batch, done_batch, disch_batch)
                        loss_list_fqe_con[m].append(loss_con)
                        
                        # Save FQE constraint model state
                        if len(self.fqe_con_models_history[m]) >= 100:
                            self.fqe_con_models_history[m].pop(0)
                            
                        self.fqe_con_models_history[m].append({
                            'update_num': model_update_counter,
                            'epoch': k,
                            'step': j,
                            'model_state': self._get_model_state(agent_fqe_con_list[m])
                        })
                        
                        con_est_value, con_est_value_up = agent_fqe_con_list[m].avg_Q_value_est(state_batch_val)
                        fqe_est_con[m].append(con_est_value)
                        
                        lambda_update_list[m] = con_est_value_up - self.cfg.constraint_limit[m]
                        lambda_t_list[m] = lambda_t_list[m] + (self.cfg.lr_lam[m] * lambda_update_list[m])
                        lambda_t_list[m] = max(0, lambda_t_list[m])
                    
                    fqi_est_value = agent_fqi.avg_Q_value_est(state_batch_val)
                    avg_q_value_obj, avg_q_value_obj_up = agent_fqe_obj.avg_Q_value_est(state_batch_val)

                    fqi_est_list.append(fqi_est_value)
                    fqe_est_obj.append(avg_q_value_obj)
                ######################################################################################
                if j % self.cfg.target_update == 0:

                    ### update the target agent for learning agent (FQI)
                    for target_param, policy_param in zip(agent_fqi.target_net.parameters(), agent_fqi.policy_net.parameters()):
                        target_param.data.copy_(self.cfg.tau * policy_param.data + (1 - self.cfg.tau) * target_param.data)

                    ### update the target agent for evaluation agent (FQE objective cost)
                    for target_param, policy_param in zip(agent_fqe_obj.target_net.parameters(), agent_fqe_obj.policy_net.parameters()):
                        target_param.data.copy_(self.cfg.tau * policy_param.data + (1 - self.cfg.tau) * target_param.data)

                    ### update the target agent for evaluation agent (FQE constraint cost)
                    for agent_fqe_con in agent_fqe_con_list:
                        for target_param, policy_param in zip(agent_fqe_con.target_net.parameters(), agent_fqe_con.policy_net.parameters()):
                            target_param.data.copy_(self.cfg.tau * policy_param.data + (1 - self.cfg.tau) * target_param.data)
                #########################################################################################
            print(f"Epoch {k + 1}/{self.cfg.train_eps}")
            print(f"Average FQE estimated objective cost after epoch {k + 1}: {np.mean(fqe_est_obj)}")        

            for m in range(len(agent_fqe_con_list)):
                print(f"Average FQE estimated constraint cost of constraint {m} after epoch {k + 1}: {np.mean(fqe_est_con[m])}")
                print(f"Dual variable of constraint {m} after epoch {k + 1}: {lambda_t_list[m]}")
                print(f"Dual variable update of constraint {m} after epoch {k + 1}: {lambda_update_list[m]}")

                self.lambda_dict[m].append(lambda_t_list[m])
                self.FQE_loss_con[m].append(np.mean(loss_list_fqe_con[m]))
                self.FQE_est_con_costs[m].append(np.mean(fqe_est_con[m]))

            self.FQI_loss.append(np.mean(loss_list_fqi))
            self.FQE_loss_obj.append(np.mean(loss_list_fqe_obj))

            self.FQI_est_values.append(np.mean(fqi_est_list))
            self.FQE_est_obj_costs.append(np.mean(fqe_est_obj))

        print("Complete Training!")
        
        # Print summary of saved models
        # print(f"Total FQE objective models saved: {len(self.fqe_obj_models_history)}")
        # for m in range(len(agent_fqe_con_list)):
        #     print(f"Total FQE constraint {m} models saved: {len(self.fqe_con_models_history[m])}")
        
        # Save the models to disk if needed
        # self._save_models_to_disk()

        return self.FQI_loss, self.FQE_loss_obj, self.FQE_loss_con, self.FQI_est_values, self.FQE_est_obj_costs, self.FQE_est_con_costs, self.lambda_dict
    
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
        for idx, model_data in enumerate(self.fqe_obj_models_history[-100:]):
            torch.save(
                model_data['model_state'], 
                f'saved_models/fqe_obj/model_{model_data["update_num"]}.pt'
            )
        
        # Save constraint FQE models
        for con_idx in self.fqe_con_models_history.keys():
            os.makedirs(f'saved_models/fqe_con_{con_idx}', exist_ok = True)
            
            for idx, model_data in enumerate(self.fqe_con_models_history[con_idx][-100:]):
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
    def __init__(self, cfg, state_id_table, rl_cont_state_table, rl_cont_state_table_scaled, state_id_table_1, rl_cont_state_table_scaled_1, terminal_state):
        self.cfg = cfg
    
        # Load datasets
        self.state_df_id = state_id_table
        self.rl_cont_state_table = rl_cont_state_table
        self.rl_cont_state_table_scaled = rl_cont_state_table_scaled

        self.state_df_id_1 = state_id_table_1
        self.rl_cont_state_table_scaled_1 = rl_cont_state_table_scaled_1

        self.terminal_state = terminal_state

    def data_buffer_test(self, num_constraint = 2):
        self.test_memory = ReplayBuffer(self.cfg.memory_capacity)

        for i in range(len(self.state_df_id)):
            state = self.rl_cont_state_table_scaled.values[i]
            action = self.state_df_id['discharge_action'].values[i]
            
            if action == 1.0:
                if self.state_df_id['death'].values[i] == 1.0:
                    if self.state_df_id['discharge_fail'].values[i] == 1.0:
                        done = 0.0
                    else:
                        done = 1.0
                else:
                    if self.state_df_id['discharge_fail'].values[i] == 0.0:
                        done = 1.0
                    else:
                        done = 0.0
            else:
                done = 0.0
            
            obj_cost = self.state_df_id['mortality_costs_md'].values[i]
            con_cost = []
            
            for j in range(num_constraint):
                cost_col = f'con_cost_{j}'  
                if cost_col in self.state_df_id.columns:
                    con_cost.append(self.state_df_id[cost_col].values[i])
                else:
                    con_cost.append(0.0) 

            if done == 0.0:
                idx = self.state_df_id.index[i]
                next_state = self.rl_cont_state_table_scaled_1.loc[idx + 1].values
            else:
                next_state = self.terminal_state

            self.test_memory.push(state, action, obj_cost, con_cost, next_state, done)

    def data_torch_loader_test(self):
        state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch = self.test_memory.extract()

        state_batch = torch.tensor(np.array(state_batch), device = self.cfg.device, dtype = torch.float)

        disch_batch = list(action_batch)
        action_batch = torch.tensor(np.array(action_batch), device = self.cfg.device, dtype = torch.long).unsqueeze(1)
        
        obj_cost_batch = torch.tensor(np.array(obj_cost_batch), device = self.cfg.device, dtype = torch.float)
        con_cost_batch = [torch.tensor(np.array(cost), device = self.cfg.device, dtype=torch.float) for cost in con_cost_batch]
        next_state_batch = torch.tensor(np.array(next_state_batch), device = self.cfg.device, dtype = torch.float)

        done_batch = torch.tensor(np.array(done_batch), device = self.cfg.device, dtype = torch.float)
        disch_batch = torch.tensor(np.array(disch_batch), device = self.cfg.device, dtype = torch.float)

        return state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch, disch_batch
    
class TestConfig:
    def __init__(self, constraint_num):
        
        self.constraint_num = constraint_num

        self.memory_capacity = int(2e6)  # capacity of Replay Memory

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check GPU
        # self.device = torch.device("cpu")