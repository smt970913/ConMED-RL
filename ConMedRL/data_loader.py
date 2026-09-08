import sys
import os

import numpy as np
import pandas as pd

import torch

script_dir = os.path.dirname(__file__)
sys.path.append(script_dir)

import conmedrl

import warnings
warnings.filterwarnings('ignore')

ROW_INDEX_COLUMN = "row_index"


def _action_value(outcome_df, i, action_name):
    """Read a scalar discrete action or a continuous action vector."""
    if isinstance(action_name, (list, tuple)):
        return outcome_df[list(action_name)].iloc[i].to_numpy(dtype=np.float32)
    return outcome_df[action_name].iloc[i]


def _done_value(outcome_df, i, action, done_condition=None):
    """Return the recorded episode boundary when the table provides one."""
    if "done" in outcome_df.columns:
        return float(outcome_df["done"].iloc[i])
    if callable(done_condition):
        return float(done_condition(action, outcome_df, i))
    if np.asarray(action).ndim > 0:
        raise ValueError(
            "Continuous/vector actions require an explicit 'done' column or "
            "done_condition; a non-zero dose is not necessarily terminal."
        )
    return float(action)


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
        self.train_memory = conmedrl.ReplayBuffer(
            self.cfg.memory_capacity,
            seed=getattr(self.cfg, "random_seed", None),
        )
        self.action_type = (
            "continuous" if isinstance(action_name, (list, tuple)) else "discrete"
        )

        for i in range(len(self.outcome_df)):
            state = self.state_var_df.values[i]
            action = _action_value(self.outcome_df, i, action_name)

            done = _done_value(self.outcome_df, i, action, done_condition)
            
            obj_cost = self.outcome_df['obj_cost'].values[i]
            con_cost = []
            
            for j in range(num_constraint):
                cost_col = f'con_cost_{j}'  
                if cost_col in self.outcome_df.columns:
                    con_cost.append(self.outcome_df[cost_col].values[i])
                else:
                    con_cost.append(0.0) 

            if done == 0.0:
                if ROW_INDEX_COLUMN in self.outcome_df.columns:
                    idx = int(self.outcome_df[ROW_INDEX_COLUMN].iloc[i])
                else:
                    idx = int(self.outcome_df.index[i])
                next_state = (
                    self.state_var_df.iloc[idx + 1].values
                    if idx + 1 < len(self.state_var_df)
                    else self.terminal_state
                )
            else:
                next_state = self.terminal_state

            self.train_memory.push(state, action, obj_cost, con_cost, next_state, done)

    def data_torch_loader_train(self):
        state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch = self.train_memory.sample(self.cfg.batch_size)

        state_batch = torch.tensor(np.array(state_batch), device = self.cfg.device, dtype = torch.float)
        if getattr(self, "action_type", "discrete") == "continuous":
            action_batch = torch.tensor(
                np.asarray(action_batch), device=self.cfg.device, dtype=torch.float
            )
            if action_batch.ndim == 1:
                action_batch = action_batch.unsqueeze(1)
        else:
            action_batch = torch.tensor(
                np.asarray(action_batch), device=self.cfg.device, dtype=torch.long
            ).unsqueeze(1)
        
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
        self.buffer_memory = conmedrl.ReplayBuffer(
            self.cfg.memory_capacity,
            seed=getattr(self.cfg, "random_seed", None),
        )
        self.action_type = (
            "continuous" if isinstance(action_name, (list, tuple)) else "discrete"
        )

        for i in range(len(self.outcome_df_select)):
            state = self.state_var_df_select.values[i]
            action = _action_value(self.outcome_df_select, i, action_name)

            done = _done_value(
                self.outcome_df_select, i, action, done_condition
            )
            
            obj_cost = self.outcome_df_select['obj_cost'].values[i]
            con_cost = []
            
            for j in range(num_constraint):
                cost_col = f'con_cost_{j}'  
                if cost_col in self.outcome_df_select.columns:
                    con_cost.append(self.outcome_df_select[cost_col].values[i])
                else:
                    con_cost.append(0.0) 

            if done == 0.0:
                if ROW_INDEX_COLUMN in self.outcome_df_select.columns:
                    idx = int(self.outcome_df_select[ROW_INDEX_COLUMN].iloc[i])
                else:
                    idx = int(self.outcome_df_select.index[i])
                next_state = (
                    self.state_var_df.iloc[idx + 1].values
                    if idx + 1 < len(self.state_var_df)
                    else self.terminal_state
                )
            else:
                next_state = self.terminal_state

            self.buffer_memory.push(state, action, obj_cost, con_cost, next_state, done)

    def data_torch_loader(self, data_type = 'val'):
        state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch = self.buffer_memory.extract()

        state_batch = torch.tensor(np.array(state_batch), device = self.cfg.device, dtype = torch.float)
        if getattr(self, "action_type", "discrete") == "continuous":
            action_batch = torch.tensor(
                np.asarray(action_batch), device=self.cfg.device, dtype=torch.float
            )
            if action_batch.ndim == 1:
                action_batch = action_batch.unsqueeze(1)
        else:
            action_batch = torch.tensor(
                np.asarray(action_batch), device=self.cfg.device, dtype=torch.long
            ).unsqueeze(1)

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
