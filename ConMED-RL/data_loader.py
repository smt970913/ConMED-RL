import sys
import os

import numpy as np
import pandas as pd

import torch

script_dir = os.path.dirname(__file__)
sys.path.append(script_dir)

import concarerl

import warnings
warnings.filterwarnings('ignore')

class TrainDataLoader:
    def __init__(self, cfg, outcome_table, state_var_table, terminal_state):
        self.cfg = cfg
    
        # Load datasets
        self.outcome_df = outcome_table
        self.state_var_df = state_var_table

        self.terminal_state = terminal_state

    def data_buffer_train(self, action_name, done_condition = None, num_constraint = 2):
        """
        Load training data into the PyTorch sampling buffer.
        
        Args:
            action_name (str): Column name for action in outcome_df
            done_condition (callable, optional):
                Should accept the recorded done values in the outcome_df.
                If None, uses the action as the done condition.
            num_constraint (int): Number of constraint costs to extract
        """
        self.train_memory = concarerl.ReplayBuffer(self.cfg.memory_capacity)

        for i in range(len(self.outcome_df)):
            state = self.state_var_df.values[i]
            action = self.outcome_df[action_name].values[i]

            # Determine done condition
            if done_condition is not None:
                done = self.outcome_df['done'].values[i]
            else:
                done = action
            
            obj_cost = self.outcome_df['obj_cost'].values[i]
            con_cost = []
            
            for j in range(num_constraint):
                cost_col = f'con_cost_{j}'  
                if cost_col in self.outcome_df.columns:
                    con_cost.append(self.outcome_df[cost_col].values[i])
                else:
                    con_cost.append(0.0) 

            if done == 0.0:
                idx = self.outcome_df.index[i]
                next_state = self.state_var_df.loc[idx + 1].values
            else:
                next_state = self.terminal_state

            self.train_memory.push(state, action, obj_cost, con_cost, next_state, done)

    def data_torch_loader_train(self):
        state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch = self.train_memory.sample(self.cfg.batch_size)

        state_batch = torch.tensor(np.array(state_batch), device = self.cfg.device, dtype = torch.float)
        action_batch = torch.tensor(np.array(action_batch), device = self.cfg.device, dtype = torch.long).unsqueeze(1)
        
        obj_cost_batch = torch.tensor(np.array(obj_cost_batch), device = self.cfg.device, dtype = torch.float)
        con_cost_batch = [torch.tensor(np.array(cost), device = self.cfg.device, dtype=torch.float) for cost in con_cost_batch]
        next_state_batch = torch.tensor(np.array(next_state_batch), device = self.cfg.device, dtype = torch.float)
        
        done_batch = torch.tensor(np.array(done_batch), device = self.cfg.device, dtype = torch.float)

        return state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch
    
class ValTestDataLoader:
    def __init__(self, cfg, outcome_table_select, state_var_table_select, outcome_table, state_var_table, terminal_state):
        self.cfg = cfg
    
        # Load datasets
        self.outcome_df_select = outcome_table_select
        self.state_var_df_select = state_var_table_select

        self.outcome_df = outcome_table
        self.state_var_df = state_var_table

        self.terminal_state = terminal_state

    def data_buffer(self, action_name, done_condition = None, num_constraint = 2):
        """
        Load validation/test data into the PyTorch sampling buffer.
        
        Args:
            action_name (str): Column name for action in outcome_df_select
            done_condition (callable, optional): Function to determine done condition.
                Should accept (action, outcome_df, idx) and return done value.
                If None, uses the action as the done condition.
            num_constraint (int): Number of constraint costs to extract
        """
        self.buffer_memory = concarerl.ReplayBuffer(self.cfg.memory_capacity)

        for i in range(len(self.outcome_df_select)):
            state = self.state_var_df_select.values[i]
            action = self.outcome_df_select[action_name].values[i]

            # Determine done condition
            if done_condition is not None:
                # Use custom done condition function
                done = done_condition(action, self.outcome_df_select, i)
            else:
                done = action
            
            obj_cost = self.outcome_df_select['obj_cost'].values[i]
            con_cost = []
            
            for j in range(num_constraint):
                cost_col = f'con_cost_{j}'  
                if cost_col in self.outcome_df_select.columns:
                    con_cost.append(self.outcome_df_select[cost_col].values[i])
                else:
                    con_cost.append(0.0) 

            if done == 0.0:
                idx = self.outcome_df_select.index[i]
                next_state = self.state_var_df.loc[idx + 1].values
            else:
                next_state = self.terminal_state

            self.buffer_memory.push(state, action, obj_cost, con_cost, next_state, done)

    def data_torch_loader(self, data_type = 'val'):
        state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch = self.buffer_memory.extract()

        state_batch = torch.tensor(np.array(state_batch), device = self.cfg.device, dtype = torch.float)
        action_batch = torch.tensor(np.array(action_batch), device = self.cfg.device, dtype = torch.long).unsqueeze(1)

        obj_cost_batch = torch.tensor(np.array(obj_cost_batch), device = self.cfg.device, dtype = torch.float)
        con_cost_batch = [torch.tensor(np.array(cost), device = self.cfg.device, dtype=torch.float) for cost in con_cost_batch]
        next_state_batch = torch.tensor(np.array(next_state_batch), device = self.cfg.device, dtype = torch.float)
        
        done_batch = torch.tensor(np.array(done_batch), device = self.cfg.device, dtype = torch.float)

        if data_type == 'val':
            return state_batch
        elif data_type == 'test':
            return state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch
        else:
            raise ValueError(f"Invalid data type: {data_type}")
