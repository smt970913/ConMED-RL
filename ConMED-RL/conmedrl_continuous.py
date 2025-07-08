import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np
from tqdm import tqdm
import copy
import os
import json

import random
import sys
import datetime
import math
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

curr_path = str(Path().absolute())
parent_path = str(Path().absolute().parent)
sys.path.append(parent_path) # add current terminal path to sys.path
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # obtain current time

class FCN_fqe(nn.Module):
    """
    Fully Connected Network for Fitted Q Evaluation (FQE).
    
    A neural network class that supports multiple activation functions with their parameters
    for Q-value estimation in offline reinforcement learning.
    
    Args:
        input_dim (int): state_dim + action_dim
        output_dim (int): 1
        hidden_layers (Optional[List[int]]): List of hidden layer sizes. If None, creates single linear layer
        activation_function (str): Name of activation function to use. Default is 'relu'
        activation_params (Optional[Dict[str, Any]]): Parameters for the activation function
    """
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 hidden_layers: Optional[List[int]] = None, 
                 activation_function: str = 'relu', 
                 activation_params: Optional[Dict[str, Any]] = None) -> None:
        
        super(FCN_fqe, self).__init__()
        self.activation_function = activation_function
        self.activation_params = activation_params if activation_params is not None else {}
        
        # Define supported activation functions and their default parameters
        self.activation_defaults: Dict[str, Dict[str, Any]] = {
            'relu': {},
            'leaky_relu': {'negative_slope': 0.01},
            'elu': {'alpha': 1.0},
            'selu': {},
            'gelu': {},
            'tanh': {},
            'sigmoid': {},
            'softplus': {'beta': 1.0, 'threshold': 20},
            'prelu': {'num_parameters': 1}  # Note: PReLU needs to be handled differently as it's a module
        }
        
        # Merge default parameters with user-provided parameters
        if self.activation_function in self.activation_defaults:
            final_params = self.activation_defaults[self.activation_function].copy()
            final_params.update(self.activation_params)
            self.activation_params = final_params

        if hidden_layers is None:
            self.fc1 = nn.Linear(input_dim, output_dim)
            self.layers = None
        else:
            layers = []
            # add the input layer
            layers.append(nn.Linear(input_dim, hidden_layers[0]))

            # add the hidden layers
            if len(hidden_layers) > 1:
                for i in range(1, len(hidden_layers)):
                    layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))

                # add the output layer
                layers.append(nn.Linear(hidden_layers[-1], output_dim))

                # store the layers in the ModuleList
                self.layers = nn.ModuleList(layers)
            else:
                # add the output layer
                layers.append(nn.Linear(hidden_layers[-1], output_dim))
                self.layers = nn.ModuleList(layers)
        
        # Handle PReLU separately as it's a learnable module
        if self.activation_function == 'prelu':
            self.prelu = nn.PReLU(num_parameters = self.activation_params.get('num_parameters', 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, state_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, action_dim)
        """
        if self.layers is None:
            x = self.fc1(x)
        else:
            # pass the input through the layers
            for layer in self.layers[:-1]:
                x = layer(x)
                x = self._apply_activation(x)

            # pass the output layer
            x = self.layers[-1](x)

        return x
    
    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the specified activation function with its parameters.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after applying activation function
            
        Raises:
            ValueError: If activation function is not supported
        """
        if self.activation_function == 'relu':
            return F.relu(x)
        elif self.activation_function == 'leaky_relu':
            return F.leaky_relu(x, negative_slope = self.activation_params['negative_slope'])
        elif self.activation_function == 'elu':
            return F.elu(x, alpha=self.activation_params['alpha'])
        elif self.activation_function == 'selu':
            return F.selu(x)
        elif self.activation_function == 'gelu':
            return F.gelu(x)
        elif self.activation_function == 'tanh':
            return F.tanh(x)
        elif self.activation_function == 'sigmoid':
            return F.sigmoid(x)
        elif self.activation_function == 'softplus':
            return F.softplus(x, beta=self.activation_params['beta'], threshold = self.activation_params['threshold'])
        elif self.activation_function == 'prelu':
            return self.prelu(x)
        else:
            raise ValueError(f"Activation function {self.activation_function} not supported")
    
    @classmethod
    def get_supported_activations(cls) -> Dict[str, str]:
        """
        Return a dictionary of supported activation functions and their parameters.
        
        Returns:
            Dict[str, str]: Dictionary mapping activation function names to parameter descriptions
        """
        return {
            'relu': 'No parameters required',
            'leaky_relu': 'negative_slope (default: 0.01)',
            'elu': 'alpha (default: 1.0)',
            'selu': 'No parameters required',
            'gelu': 'No parameters required',
            'tanh': 'No parameters required',
            'sigmoid': 'No parameters required',
            'softplus': 'beta (default: 1.0), threshold (default: 20)',
            'prelu': 'num_parameters (default: 1)'
        }

class FCN_critic(nn.Module):
    """
    Fully Connected Network for Critic.
    
    A neural network class that supports multiple activation functions with their parameters
    for Q-value estimation in offline reinforcement learning policy optimization.
    
    Args:
        input_dim (int): state_dim + action_dim
        output_dim (int): 1
        hidden_layers (Optional[List[int]]): List of hidden layer sizes. If None, creates single linear layer
        activation_function (str): Name of activation function to use. Default is 'relu'
        activation_params (Optional[Dict[str, Any]]): Parameters for the activation function
    """
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 hidden_layers: Optional[List[int]] = None, 
                 activation_function: str = 'relu', 
                 activation_params: Optional[Dict[str, Any]] = None) -> None:
        
        super(FCN_critic, self).__init__()
        self.activation_function = activation_function
        self.activation_params = activation_params if activation_params is not None else {}

        # Define supported activation functions and their default parameters
        self.activation_defaults: Dict[str, Dict[str, Any]] = {
            'relu': {},
            'leaky_relu': {'negative_slope': 0.01},
            'elu': {'alpha': 1.0},
            'selu': {},
            'gelu': {},
            'tanh': {},
            'sigmoid': {},
            'softplus': {'beta': 1.0, 'threshold': 20},
            'prelu': {'num_parameters': 1}  # Note: PReLU needs to be handled differently as it's a module
        }

        # Merge default parameters with user-provided parameters
        if self.activation_function in self.activation_defaults:
            final_params = self.activation_defaults[self.activation_function].copy()
            final_params.update(self.activation_params)
            self.activation_params = final_params

        if hidden_layers is None:
            self.fc1 = nn.Linear(input_dim, output_dim)
            self.layers = None
        else:
            # initialize the list of layers
            layers = []

            # add the input layer
            layers.append(nn.Linear(input_dim, hidden_layers[0]))

            # add the hidden layers
            if len(hidden_layers) > 1:
                for i in range(1, len(hidden_layers)):
                    layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))

                # add the output layer
                layers.append(nn.Linear(hidden_layers[-1], output_dim))

                # store the layers in the ModuleList
                self.layers = nn.ModuleList(layers)
            else:
                # add the output layer
                layers.append(nn.Linear(hidden_layers[-1], output_dim))
                self.layers = nn.ModuleList(layers)
        
        # Handle PReLU separately as it's a learnable module
        if self.activation_function == 'prelu':
            self.prelu = nn.PReLU(num_parameters = self.activation_params.get('num_parameters', 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, state_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, action_dim)
        """
        if self.layers is None:
            x = self.fc1(x)
        else:
            # pass the input through the layers
            for layer in self.layers[:-1]:
                x = layer(x)
                x = self._apply_activation(x)

            # pass the output layer
            x = self.layers[-1](x)

        return x
    
    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the specified activation function with its parameters.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after applying activation function
            
        Raises:
            ValueError: If activation function is not supported
        """
        if self.activation_function == 'relu':
            return F.relu(x)
        elif self.activation_function == 'leaky_relu':
            return F.leaky_relu(x, negative_slope = self.activation_params['negative_slope'])
        elif self.activation_function == 'elu':
            return F.elu(x, alpha=self.activation_params['alpha'])
        elif self.activation_function == 'selu':
            return F.selu(x)
        elif self.activation_function == 'gelu':
            return F.gelu(x)
        elif self.activation_function == 'tanh':
            return F.tanh(x)
        elif self.activation_function == 'sigmoid':
            return F.sigmoid(x)
        elif self.activation_function == 'softplus':
            return F.softplus(x, beta=self.activation_params['beta'], threshold = self.activation_params['threshold'])
        elif self.activation_function == 'prelu':
            return self.prelu(x)
        else:
            raise ValueError(f"Activation function {self.activation_function} not supported")

    @classmethod
    def get_supported_activations(cls) -> Dict[str, str]:
        """
        Return a dictionary of supported activation functions and their parameters.
        
        Returns:
            Dict[str, str]: Dictionary mapping activation function names to parameter descriptions
        """
        return {
            'relu': 'No parameters required',
            'leaky_relu': 'negative_slope (default: 0.01)',
            'elu': 'alpha (default: 1.0)',
            'selu': 'No parameters required',
            'gelu': 'No parameters required',
            'tanh': 'No parameters required',
            'sigmoid': 'No parameters required',
            'softplus': 'beta (default: 1.0), threshold (default: 20)',
            'prelu': 'num_parameters (default: 1)'
        }
    
class FCN_actor(nn.Module):
    """
    Fully Connected Network for Actor.
    
    A neural network class that supports multiple activation functions with their parameters
    for action selection in offline reinforcement learning policy optimization.

    Args:
        input_dim (int): state_dim
        output_dim (int): action_dim
        hidden_layers (Optional[List[int]]): List of hidden layer sizes. If None, creates single linear layer
        activation_function (str): Name of activation function to use. Default is 'relu'
        activation_params (Optional[Dict[str, Any]]): Parameters for the activation function
    """

    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 hidden_layers: Optional[List[int]] = None, 
                 activation_function: str = 'relu', 
                 activation_params: Optional[Dict[str, Any]] = None) -> None:
        
        super(FCN_actor, self).__init__()
        self.activation_function = activation_function
        self.activation_params = activation_params if activation_params is not None else {}

        # Define supported activation functions and their default parameters
        self.activation_defaults: Dict[str, Dict[str, Any]] = {
            'relu': {},
            'leaky_relu': {'negative_slope': 0.01},
            'elu': {'alpha': 1.0},
            'selu': {},
            'gelu': {},
            'tanh': {},
            'sigmoid': {},
            'softplus': {'beta': 1.0, 'threshold': 20},
            'prelu': {'num_parameters': 1}  # Note: PReLU needs to be handled differently as it's a module
        }

        # Merge default parameters with user-provided parameters
        if self.activation_function in self.activation_defaults:
            final_params = self.activation_defaults[self.activation_function].copy()
            final_params.update(self.activation_params)
            self.activation_params = final_params

        if hidden_layers is None:
            self.fc1 = nn.Linear(input_dim, output_dim)
            self.layers = None
        else:
            # initialize the list of layers
            layers = []

            # add the input layer
            layers.append(nn.Linear(input_dim, hidden_layers[0]))

            # add the hidden layers
            if len(hidden_layers) > 1:
                for i in range(1, len(hidden_layers)):
                    layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))

                # add the output layer
                layers.append(nn.Linear(hidden_layers[-1], output_dim))

                # store the layers in the ModuleList
                self.layers = nn.ModuleList(layers)
            else:
                # add the output layer
                layers.append(nn.Linear(hidden_layers[-1], output_dim))
                self.layers = nn.ModuleList(layers)
        
        # Handle PReLU separately as it's a learnable module
        if self.activation_function == 'prelu':
            self.prelu = nn.PReLU(num_parameters = self.activation_params.get('num_parameters', 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, state_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, action_dim)
        """
        if self.layers is None:
            x = self.fc1(x)
        else:
            # pass the input through the layers
            for layer in self.layers[:-1]:
                x = layer(x)
                x = self._apply_activation(x)

            # pass the output layer
            x = self.layers[-1](x)

        return x
    
    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the specified activation function with its parameters.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after applying activation function
            
        Raises:
            ValueError: If activation function is not supported
        """
        if self.activation_function == 'relu':
            return F.relu(x)
        elif self.activation_function == 'leaky_relu':
            return F.leaky_relu(x, negative_slope = self.activation_params['negative_slope'])
        elif self.activation_function == 'elu':
            return F.elu(x, alpha=self.activation_params['alpha'])
        elif self.activation_function == 'selu':
            return F.selu(x)
        elif self.activation_function == 'gelu':
            return F.gelu(x)
        elif self.activation_function == 'tanh':
            return F.tanh(x)
        elif self.activation_function == 'sigmoid':
            return F.sigmoid(x)
        elif self.activation_function == 'softplus':
            return F.softplus(x, beta=self.activation_params['beta'], threshold = self.activation_params['threshold'])
        elif self.activation_function == 'prelu':
            return self.prelu(x)
        else:
            raise ValueError(f"Activation function {self.activation_function} not supported")

    @classmethod
    def get_supported_activations(cls) -> Dict[str, str]:
        """
        Return a dictionary of supported activation functions and their parameters.
        
        Returns:
            Dict[str, str]: Dictionary mapping activation function names to parameter descriptions
        """
        return {
            'relu': 'No parameters required',
            'leaky_relu': 'negative_slope (default: 0.01)',
            'elu': 'alpha (default: 1.0)',
            'selu': 'No parameters required',
            'gelu': 'No parameters required',
            'tanh': 'No parameters required',
            'sigmoid': 'No parameters required',
            'softplus': 'beta (default: 1.0), threshold (default: 20)',
            'prelu': 'num_parameters (default: 1)'
        }


# Define the replay buffer class
class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    
    Stores transitions in the form (state, action, objective_cost, constraint_cost, next_state, done)
    and provides methods for adding experiences and sampling batches for training.
    
    Args:
        capacity (int): Maximum number of transitions to store
    """
    
    def __init__(self, capacity: int) -> None:
        """
        Initialize the replay buffer.
        
        Args:
            capacity (int): Maximum capacity of the buffer
        """
        self.capacity = capacity
        self.buffer: List[Tuple[Any, Any, Any, List[Any], Any, Any]] = []
        self.position = 0

    def push(self, 
             state: Any, 
             action: Any, 
             obj_cost: Any, 
             con_cost: Union[List[Any], Any], 
             next_state: Any, 
             done: Any) -> None:
        """
        Add a new transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            obj_cost: Objective cost/reward
            con_cost: Constraint cost(s) - can be single value or list
            next_state: Next state
            done: Whether episode is done
        """
        if not isinstance(con_cost, list) and not isinstance(con_cost, tuple):
            con_cost = [con_cost]

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state, action, obj_cost, con_cost, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[Tuple[Any, ...], Tuple[Any, ...], Tuple[Any, ...], List[List[Any]], Tuple[Any, ...], Tuple[Any, ...]]:
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            Tuple containing batched states, actions, objective costs, constraint costs, next states, and done flags
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, obj_cost, con_cost, next_state, done = zip(*batch)

        con_cost = [list(costs) for costs in zip(*con_cost)]

        return state, action, obj_cost, con_cost, next_state, done

    def extract(self) -> Tuple[Tuple[Any, ...], Tuple[Any, ...], Tuple[Any, ...], List[List[Any]], Tuple[Any, ...], Tuple[Any, ...]]:
        """
        Extract all transitions from the buffer.
        
        Returns:
            Tuple containing all states, actions, objective costs, constraint costs, next states, and done flags
        """
        batch = self.buffer
        state, action, obj_cost, con_cost, next_state, done = zip(*batch)

        con_cost = [list(costs) for costs in zip(*con_cost)]

        return state, action, obj_cost, con_cost, next_state, done

    def clear(self) -> None:
        """Clear the buffer and reset position."""
        self.buffer = []
        self.position = 0

    def __len__(self) -> int:
        """
        Get the current size of the buffer.
        
        Returns:
            int: Number of transitions currently stored
        """
        return len(self.buffer)

class FQE:
    """
    Fitted Q Evaluation (FQE) for policy evaluation in offline constrained reinforcement learning.
    
    FQE estimates the Q-values of a given policy using offline data without policy interaction.
    It can evaluate both objective costs and constraint costs.
    
    Args:
        cfg: Configuration object containing hyperparameters
        input_dim (int): state_dim + action_dim
        output_dim (int): 1
        hidden_layers (Optional[List[int]]): List of hidden layer sizes
        weight_decay (float): Weight decay for regularization
        eval_agent: Agent whose policy is being evaluated (Actor)
        eval_target (str): Target to evaluate ('obj' for objective, constraint index for constraints)
    """
    
    def __init__(self, 
                 cfg: Any, 
                 input_dim: int, 
                 output_dim: 1, 
                 hidden_layers: Optional[List[int]], 
                 weight_decay: float, 
                 eval_agent: Any, 
                 eval_target: Union[str, int] = 'obj') -> None:
        """
        Initialize the FQE agent.
        
        Args:
            cfg: Configuration object
            input_dim (int): state_dim + action_dim
            output_dim (int): 1
            hidden_layers (Optional[List[int]]): Hidden layer sizes
            weight_decay (float): Weight decay value
            eval_agent: Policy agent to evaluate
            eval_target (Union[str, int]): Evaluation target
        """
        self.device = cfg.device
        self.gamma = cfg.gamma  # discount factor
        self.weight_decay = weight_decay

        if eval_target == 'obj':
            self.lr_fqe = cfg.lr_fqe_obj
        else:
            self.lr_fqe = cfg.lr_fqe_con[eval_target]  # For constraint cost, specify which constraint to evaluate

        # define policy Q-Estimator
        self.policy_net = FCN_fqe(input_dim, output_dim, hidden_layers, 
                                  cfg.activation_function_fqe, cfg.activation_params_fqe).to(self.device)
        # define target Q-Estimator
        self.target_net = FCN_fqe(input_dim, output_dim, hidden_layers, 
                                  cfg.activation_function_fqe, cfg.activation_params_fqe).to(self.device)

        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        if hidden_layers is None:
            self.optimizer = optim.SGD(self.policy_net.parameters(), lr = self.lr_fqe)
        else:
            self.optimizer = cfg.optimizer_fqe(self.policy_net.parameters(), 
                                               lr = self.lr_fqe, 
                                               weight_decay = self.weight_decay) 

        self.loss = cfg.loss_fqe  # loss function

        # input the evaluation agent - Actor
        self.eval_agent = eval_agent

    def update(self, 
               state_action_batch: torch.Tensor, 
               cost_batch: torch.Tensor, 
               next_state_batch: torch.Tensor, 
               done_batch: torch.Tensor) -> float:
        """
        Update the Q-network using a batch of transitions.
        
        Args:
            state_action_batch (torch.Tensor): Batch of states and actions
            cost_batch (torch.Tensor): Batch of costs/rewards
            next_state_batch (torch.Tensor): Batch of next states
            done_batch (torch.Tensor): Batch of done flags
            
        Returns:
            float: Loss value after update
        """
        # We need to evaluate the parameterized policy derived by the actor
        policy_action_batch = self.eval_agent.rl_policy(next_state_batch, use_para = 'policy')

        # predicted Q-value using policy Q-network
        q_values = self.policy_net(state_action_batch)

        # target Q-value calculated by target Q-network
        next_state_action_batch = torch.cat((next_state_batch, policy_action_batch), dim = 1)
        next_q_values = self.target_net(next_state_action_batch)

        expected_q_values = cost_batch + self.gamma * next_q_values * (1 - done_batch)

        loss = self.loss(q_values, expected_q_values.unsqueeze(1))

        # Update Q-network by minimizing the above loss function
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()
    
    def avg_Q_value_est(self, state_batch: torch.Tensor, z_value: float) -> Tuple[float, float]:
        """
        Estimate average Q-value and confidence bound for a batch of states.
        
        Args:
            state_batch (torch.Tensor): Batch of states
            z_value (float): Z-score for confidence interval
            
        Returns:
            Tuple[float, float, float]: Mean Q-value, upper confidence bound, and lower confidence bound
        """
        policy_action_batch = self.eval_agent.rl_policy(state_batch, use_para = 'policy')

        state_action_batch = torch.cat((state_batch, policy_action_batch), dim = 1)
        
        q_values = self.policy_net(state_action_batch)

        q_mean = q_values.mean()
        q_std = q_values.std()
        n = q_values.shape[0]
    
        if n <= 1 or q_std == 0:
            return q_mean.item(), q_mean.item()
    
        z = z_value  
        q_upper_bound = q_mean + z * (q_std / math.sqrt(n))
        q_lower_bound = q_mean - z * (q_std / math.sqrt(n))
    
        return q_mean.item(), q_upper_bound.item(), q_lower_bound.item()

    def save(self, path: str) -> None:
        """
        Save the trained networks.
        
        Args:
            path (str): Directory path to save the models
        """
        torch.save(self.policy_net.state_dict(), path + 'FQE_policy_network.pth')
        torch.save(self.target_net.state_dict(), path + 'FQE_target_network.pth')

class Critic:
    """
    Critic for constrained policy optimization in offline constrained reinforcement learning.
    
    Critic learns the Q-value function of the state-action pairs.
    It handles both objective costs and constraint costs through Lagrangian methods.
    
    Args:
        cfg: Configuration object containing hyperparameters
        input_dim (int): state_dim + action_dim
        output_dim (int): 1
        hidden_layers (Optional[List[int]]): List of hidden layer sizes
        weight_decay (float): Weight decay for regularization
    """
    
    def __init__(self, 
                 cfg: Any, 
                 input_dim: int, 
                 output_dim: 1, 
                 hidden_layers: Optional[List[int]], 
                 weight_decay: float) -> None:
        """
        Initialize the Critic agent.
        
        Args:
            cfg: Configuration object
            input_dim (int): state_dim + action_dim
            output_dim (int): 1
            hidden_layers (Optional[List[int]]): Hidden layer sizes
            weight_decay (float): Weight decay value
        """
        self.device = cfg.device
        self.gamma = cfg.gamma
        self.weight_decay = weight_decay

        self.policy_net = FCN_critic(input_dim, output_dim, hidden_layers,
                                  cfg.activation_function_critic, cfg.activation_params_critic).to(self.device)
        self.target_net = FCN_critic(input_dim, output_dim, hidden_layers,
                                 cfg.activation_function_critic, cfg.activation_params_critic).to(self.device)

        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        if hidden_layers is None:
            self.optimizer = optim.SGD([
                {'params': self.policy_net.fc1.weight, 'weight_decay': self.weight_decay}, 
                {'params': self.policy_net.fc1.bias,   'weight_decay': 0.0}
                ], lr = cfg.lr_critic)
        else:
            self.optimizer = cfg.optimizer_critic(self.policy_net.parameters(), lr = cfg.lr_critic, weight_decay = self.weight_decay)
        
        self.loss = cfg.loss_critic

    def update(self, 
               actor_agent: Any,
               lambda_t_list: List[float], 
               state_action_batch: torch.Tensor, 
               obj_cost_batch: torch.Tensor, 
               con_cost_batch: List[torch.Tensor], 
               next_state_batch: torch.Tensor, 
               done_batch: torch.Tensor) -> float:
        """
        Update the Q-network using a batch of transitions with Lagrangian constraints.
        
        Args:
            actor_agent (Any): The Actor agent to be used
            lambda_t_list (List[float]): List of Lagrange multipliers for constraints
            state_action_batch (torch.Tensor): Batch of states and actions
            obj_cost_batch (torch.Tensor): Batch of objective costs
            con_cost_batch (List[torch.Tensor]): List of constraint cost batches
            next_state_batch (torch.Tensor): Batch of next states
            done_batch (torch.Tensor): Batch of done flags
            
        Returns:
            float: Loss value after update
        """
        self.actor_agent = actor_agent

        q_values = self.policy_net(state_action_batch)
        next_action_batch = self.actor_agent.rl_policy(next_state_batch, use_para = 'target')
        next_state_action_batch = torch.cat((next_state_batch, next_action_batch), dim = 1)
        next_q_values = self.target_net(next_state_action_batch)

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
           
    def avg_Q_value_est(self, state_batch: torch.Tensor) -> float:
        """
        Estimate average Q-value for a batch of states using the learned policy.
        
        Args:
            state_batch (torch.Tensor): Batch of states
            
        Returns:
            float: Average Q-value
        """
        action_pred_batch = self.actor_agent.rl_policy(state_batch, use_para = 'policy')
        state_action_batch = torch.cat((state_batch, action_pred_batch), dim = 1)
        q_values = self.policy_net(state_action_batch)
        avg_q_values = q_values.mean().item()

        return avg_q_values

    def save(self, path: str) -> None:
        """
        Save the trained networks.
        
        Args:
            path (str): Directory path to save the models
        """
        torch.save(self.policy_net.state_dict(), path + 'Offline_Critic_policy_network.pth')
        torch.save(self.target_net.state_dict(), path + 'Offline_Critic_target_network.pth')

class Actor:
    """
    Actor for constrained policy optimization in offline constrained reinforcement learning.
    
    Actor learns an optimal policy using offline data.
    It handles both objective costs and constraint costs through Lagrangian methods.
    
    Args:
        cfg: Configuration object containing hyperparameters
        input_dim (int): state_dim
        output_dim (int): selected action dimension
        hidden_layers (Optional[List[int]]): List of hidden layer sizes
        weight_decay (float): Weight decay for regularization
    """
    def __init__(self, 
                 cfg: Any, 
                 input_dim: int, 
                 output_dim: int, 
                 hidden_layers: Optional[List[int]], 
                 weight_decay: float,
                 critic_agent: Any) -> None:
        """
        Initialize the Actor agent.
        
        Args:
            cfg: Configuration object
            input_dim (int): state_dim
            output_dim (int): selected action dimension
            hidden_layers (Optional[List[int]]): Hidden layer sizes
            weight_decay (float): Weight decay value
            critic_agent: Critic agent to estimate Q-values
        """
        self.device = cfg.device
        self.gamma = cfg.gamma
        self.weight_decay = weight_decay

        self.policy_net = FCN_actor(input_dim, output_dim, hidden_layers,
                                  cfg.activation_function_actor, cfg.activation_params_actor).to(self.device)
        self.target_net = FCN_actor(input_dim, output_dim, hidden_layers,
                                 cfg.activation_function_actor, cfg.activation_params_actor).to(self.device)

        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        if hidden_layers is None:
            self.optimizer = optim.SGD([
                {'params': self.policy_net.fc1.weight, 'weight_decay': self.weight_decay}, 
                {'params': self.policy_net.fc1.bias,   'weight_decay': 0.0}
                ], lr = cfg.lr_actor)
        else:
            self.optimizer = cfg.optimizer_actor(self.policy_net.parameters(), lr = cfg.lr_actor, weight_decay = self.weight_decay)

        # input the evaluation agent - Critic
        self.critic_agent = critic_agent

    def update(self, state_batch: torch.Tensor) -> float:
        """
        Update the Actor agent using a batch of transitions.
        
        Args:
            state_batch (torch.Tensor): Batch of states
            
        Returns:
            float: Loss value after update
        """
        action_pred_batch = self.policy_net(state_batch)
        q_values = self.critic_agent.policy_net(torch.cat((state_batch, action_pred_batch), dim = 1))

        # Positive Q-value mean to minimize Q-value (gradient descent)
        loss = q_values.mean()

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()
           
    def rl_policy(self, state_batch: torch.Tensor, use_para: str = 'target') -> torch.Tensor:
        """
        Get policy actions for a batch of states.
        
        Args:
            state_batch (torch.Tensor): Batch of states
            use_para (str): 'policy' or 'target'
        Returns:
            torch.Tensor: Batch of actions selected by the policy
        """
        if use_para == 'policy':
            action_pred_batch = self.policy_net(state_batch)
        elif use_para == 'target':
            action_pred_batch = self.target_net(state_batch)

        return action_pred_batch

    def save(self, path: str) -> None:
        """
        Save the trained networks.
        
        Args:
            path (str): Directory path to save the models
        """
        torch.save(self.policy_net.state_dict(), path + 'Offline_Actor_policy_network.pth')
        torch.save(self.target_net.state_dict(), path + 'Offline_Actor_target_network.pth')

### Customized configuration settings for studying any medical decision-making problems
class RLConfig_custom:
    """
    Configuration class for Continuous Action Space Offline Constrained Reinforcement Learning.
    
    This class stores all hyperparameters and settings needed for training RL agents,
    including network architectures, optimizers, learning rates, and constraint settings.
    
    Args:
        algo_name (str): Name of the RL algorithm
        gamma (float): Discount factor for future rewards
        batch_size (int): Batch size for training
        train_eps (int): Number of training episodes
        train_eps_steps (int): Number of steps per training episode
        lr_actor (float): Learning rate for Actor network
        lr_critic (float): Learning rate for Critic network
        weight_decay_actor (float): Weight decay for Actor optimizer
        weight_decay_critic (float): Weight decay for Critic optimizer
        optim_actor (str): Optimizer name for Actor agent
        optim_critic (str): Optimizer name for Critic agent
        loss_critic (str): Loss function name for Critic agent
        activation_function_actor (str): Activation function for Actor networks
        activation_params_actor (Dict[str, Any]): Parameters for Actor activation function
        activation_function_critic (str): Activation function for Critic networks
        activation_params_critic (Dict[str, Any]): Parameters for Critic activation function
        weight_decay_fqe (float): Weight decay for FQE optimizer
        optim_fqe (str): Optimizer name for FQE agents
        loss_fqe (str): Loss function name for FQE agents
        activation_function_fqe (str): Activation function for FQE networks
        activation_params_fqe (Dict[str, Any]): Parameters for FQE activation function
        memory_capacity (int): Capacity of replay buffer
        target_update (int): Frequency of target network updates
        tau (float): Soft update parameter
        lr_fqe_obj (float): Learning rate for objective FQE agent
        lr_fqe_con_list (List[float]): Learning rates for constraint FQE agents
        lr_lambda_list (List[float]): Learning rates for Lagrange multipliers
        constraint_num (int): Number of constraints
        threshold_list (List[float]): Constraint threshold values
        device_type (str): Device type ('cuda' or 'cpu')
        lambda_update (Optional[str]): Method for updating Lagrange multipliers (None, 'PG with bound', 'EG with bound')
        bound_lambda (Optional[float]): Bound for Lagrange multipliers when using bounded updates
    """
    
    def __init__(self, 
                 algo_name: str, 
                 gamma: float, 
                 batch_size: int,
                 train_eps: int, 
                 train_eps_steps: int,
                 # Actor and Critic configurations
                 lr_actor: float,
                 lr_critic: float,
                 weight_decay_actor: float,
                 weight_decay_critic: float,
                 optim_actor: str,
                 optim_critic: str,
                 loss_critic: str,
                 activation_function_actor: str,
                 activation_params_actor: Dict[str, Any],
                 activation_function_critic: str,
                 activation_params_critic: Dict[str, Any],
                 # FQE configurations
                 weight_decay_fqe: float,
                 optim_fqe: str,
                 loss_fqe: str,
                 activation_function_fqe: str, 
                 activation_params_fqe: Dict[str, Any],
                 # General configurations
                 memory_capacity: int, 
                 target_update: int, 
                 tau: float,
                 lr_fqe_obj: float, 
                 lr_fqe_con_list: List[float], 
                 lr_lambda_list: List[float], 
                 constraint_num: int, 
                 threshold_list: List[float], 
                 device_type: str,
                 lambda_update: Optional[str] = None,
                 bound_lambda: Optional[float] = None) -> None:
        """
        Initialize the RLConfig_custom with the given parameters.
        
        Args:
            All parameters as described in class docstring
        """
        # Optimizer mapping dictionary
        self.optimizer_mapping: Dict[str, Any] = {
            'torch.optim.SGD': torch.optim.SGD,
            'torch.optim.Adam': torch.optim.Adam,
            'torch.optim.AdamW': torch.optim.AdamW,
            'torch.optim.RMSprop': torch.optim.RMSprop,
            'torch.optim.Adagrad': torch.optim.Adagrad,
        }
        
        # Loss function mapping dictionary
        self.loss_mapping: Dict[str, Any] = {
            'nn.MSELoss()': nn.MSELoss(),
            'nn.L1Loss()': nn.L1Loss(),
            'nn.SmoothL1Loss()': nn.SmoothL1Loss(),
            'nn.CrossEntropyLoss()': nn.CrossEntropyLoss(),
            'nn.BCELoss()': nn.BCELoss(),
        }
        
        self.algo = algo_name  # name of algorithm
        self.gamma = gamma # discount factor
        self.batch_size = batch_size # batch size of each training step

        self.train_eps = train_eps  # the number of training episodes
        self.train_eps_steps = train_eps_steps  # the number of steps in each training episode

        # Actor and Critic configurations
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.weight_decay_actor = weight_decay_actor
        self.weight_decay_critic = weight_decay_critic
        self.optimizer_actor = self._get_optimizer(optim_actor)
        self.optimizer_critic = self._get_optimizer(optim_critic)
        self.loss_critic = self._get_loss_function(loss_critic)
        self.activation_function_actor = activation_function_actor
        self.activation_params_actor = activation_params_actor
        self.activation_function_critic = activation_function_critic
        self.activation_params_critic = activation_params_critic

        # FQE configurations
        self.optimizer_fqe = self._get_optimizer(optim_fqe)
        self.weight_decay_fqe = weight_decay_fqe
        self.loss_fqe = self._get_loss_function(loss_fqe)
        self.activation_function_fqe = activation_function_fqe
        self.activation_params_fqe = activation_params_fqe

        self.memory_capacity = memory_capacity  # capacity of Replay Memory

        self.target_update = target_update  # update frequency of target net
        self.tau = tau # soft update parameter        

        # learning rates
        self.lr_fqe_obj = lr_fqe_obj
        self.lr_fqe_con: List[float] = [0 for i in range(constraint_num)]
        self.lr_lam: List[float] = [0 for i in range(constraint_num)]

        # constraint threshold
        self.constraint_num = constraint_num
        
        self.constraint_limit: List[float] = [0 for i in range(constraint_num)]
        for i in range(constraint_num):
            self.lr_fqe_con[i] = lr_fqe_con_list[i]
            self.lr_lam[i] = lr_lambda_list[i]
            self.constraint_limit[i] = threshold_list[i]

        # Lagrange multiplier update configuration
        self.lambda_update = lambda_update
        self.bound_lambda = bound_lambda

        if device_type == 'cuda':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check gpu
        else:
            self.device = torch.device("cpu")
    
    def _get_optimizer(self, optimizer_str: str) -> Any:
        """
        Convert optimizer string to actual optimizer class.
        
        Args:
            optimizer_str (str): String representation of optimizer
            
        Returns:
            Any: Optimizer class
        """
        if optimizer_str in self.optimizer_mapping:
            return self.optimizer_mapping[optimizer_str]
        else:
            print(f"Warning: Optimizer '{optimizer_str}' not found in mapping. Using SGD as default.")
            return torch.optim.SGD
    
    def _get_loss_function(self, loss_str: str) -> Any:
        """
        Convert loss function string to actual loss function.
        
        Args:
            loss_str (str): String representation of loss function
            
        Returns:
            Any: Loss function instance
        """
        if loss_str in self.loss_mapping:
            return self.loss_mapping[loss_str]
        else:
            print(f"Warning: Loss function '{loss_str}' not found in mapping. Using MSELoss as default.")
            return nn.MSELoss()

class RLConfigurator:
    """
    Class to configure RL settings through json file or user input.
    
    This class provides an interactive interface for users to configure all aspects
    of reinforcement learning training, including network architectures, optimizers,
    learning rates, constraints, and activation functions.
    """
    
    def __init__(self) -> None:
        """Initialize the configurator."""
        self.config: Optional[RLConfig_custom] = None

    def choose_config_method(self) -> Optional[RLConfig_custom]:
        """
        Allow user to choose between loading configuration from JSON file or manual input.
        
        Returns:
            Optional[RLConfig_custom]: Configuration object if successful, None if error occurred
        """
        print("Configuration Options:", flush = True)
        print("1. Load from JSON file (config.json)", flush = True)
        print("2. Manual input", flush = True)
        
        choice = input("Choose configuration method (1 or 2): ").strip()
        
        if choice == '1':
            return self.load_config_from_json('config.json')
        elif choice == '2':
            return self.input_rl_config()
        else:
            print("Invalid choice. Please enter 1 or 2.")
            return self.choose_config_method()

    def load_config_from_json(self, file_path: str) -> Optional[RLConfig_custom]:
        """
        Load configuration from JSON file.
        
        Args:
            file_path (str): Path to the JSON configuration file
            
        Returns:
            Optional[RLConfig_custom]: Configuration object if successful, None if error occurred
        """
        try:
            with open(file_path, 'r') as file:
                config_data = json.load(file)
            
            print(f"Loading configuration from {file_path}...", flush = True)
            print("Configuration loaded successfully!", flush = True)
            
            # Display loaded configuration
            print("\n=== Loaded Configuration ===", flush = True)
            for key, value in config_data.items():
                print(f"{key}: {value}", flush = True)
            print("=" * 30, flush = True)
            
            self.config = RLConfig_custom(**config_data)
            return self.config
        except FileNotFoundError:
            print(f"Error: Configuration file '{file_path}' not found.")
            print("Please make sure the config.json file exists in the current directory.")
            return None
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in '{file_path}': {e}")
            return None
        except Exception as e:
            print(f"Error loading configuration from JSON: {e}")
            return None

    def save_config_to_json(self, file_path: str = 'config_saved.json') -> bool:
        """
        Save current configuration to JSON file.
        
        Args:
            file_path (str): Path to save the JSON configuration file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.config is None:
            print("No configuration to save.")
            return False
            
        try:
            config_dict = {
                'algo_name': self.config.algo,
                'gamma': self.config.gamma,
                'batch_size': self.config.batch_size,
                'train_eps': self.config.train_eps,
                'train_eps_steps': self.config.train_eps_steps,
                # Actor and Critic configurations
                'lr_actor': self.config.lr_actor,
                'lr_critic': self.config.lr_critic,
                'weight_decay_actor': self.config.weight_decay_actor,
                'weight_decay_critic': self.config.weight_decay_critic,
                'optim_actor': 'torch.optim.Adam',  # You may need to store optimizer string
                'optim_critic': 'torch.optim.Adam',  # You may need to store optimizer string
                'loss_critic': 'nn.MSELoss()',  # You may need to store loss string
                'activation_function_actor': self.config.activation_function_actor,
                'activation_params_actor': self.config.activation_params_actor,
                'activation_function_critic': self.config.activation_function_critic,
                'activation_params_critic': self.config.activation_params_critic,
                # FQE configurations
                'optim_fqe': 'torch.optim.Adam',  # You may need to store optimizer string
                'weight_decay_fqe': self.config.weight_decay_fqe,
                'loss_fqe': 'nn.MSELoss()',  # You may need to store loss string
                'activation_function_fqe': self.config.activation_function_fqe,
                'activation_params_fqe': self.config.activation_params_fqe,
                # General configurations
                'memory_capacity': self.config.memory_capacity,
                'target_update': self.config.target_update,
                'tau': self.config.tau,
                'lr_fqe_obj': self.config.lr_fqe_obj,
                'constraint_num': self.config.constraint_num,
                'lr_fqe_con_list': self.config.lr_fqe_con,
                'lr_lambda_list': self.config.lr_lam,
                'threshold_list': self.config.constraint_limit,
                'device_type': 'cuda' if 'cuda' in str(self.config.device) else 'cpu',
                'lambda_update': self.config.lambda_update,
                'bound_lambda': self.config.bound_lambda
            }
            
            with open(file_path, 'w') as file:
                json.dump(config_dict, file, indent = 4)
            
            print(f"Configuration saved to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving configuration to JSON: {e}")
            return False

    def input_rl_config(self) -> Optional[RLConfig_custom]:
        """
        Gather RL configuration settings from user input.
        
        Prompts the user for all necessary hyperparameters and settings,
        validates the inputs, and creates a configuration object.
        
        Returns:
            Optional[RLConfig_custom]: Configuration object if successful, None if error occurred
        """
        try:
            algo_name = input("Enter the Algorithm Name (for training): ")
            gamma = self._get_float_input("Enter the discount factor γ: ", 0, 1)
            batch_size = self._get_int_input("Enter the batch size: ", 1)

            train_eps = self._get_int_input("Enter the number of training episodes: ", 1)
            train_eps_steps = self._get_int_input("Enter the number of steps in each training episode: ", 1)

            print("\n=== Actor and Critic Configuration ===", flush = True)
            print("\nAvailable optimizer options:", flush = True)
            print("- torch.optim.SGD", flush = True)
            print("- torch.optim.Adam", flush = True)
            print("- torch.optim.AdamW", flush = True)
            print("- torch.optim.RMSprop", flush = True)
            print("- torch.optim.Adagrad", flush = True)

            lr_actor = self._get_float_input("Enter the learning rate for Actor: ", 0)
            lr_critic = self._get_float_input("Enter the learning rate for Critic: ", 0)

            optim_actor = input("Enter the optimizer for Actor: ")
            weight_decay_actor = self._get_weight_decay(optim_actor)

            optim_critic = input("Enter the optimizer for Critic: ")
            weight_decay_critic = self._get_weight_decay(optim_critic)

            print("\nAvailable loss functions for Critic:", flush = True)
            print("- nn.MSELoss()", flush = True)
            print("- nn.L1Loss()", flush = True)
            print("- nn.SmoothL1Loss()", flush = True)
            print("- nn.CrossEntropyLoss()", flush = True)
            print("- nn.BCELoss()", flush = True)

            loss_critic = input("Enter the loss function for Critic: ")

            # Configure activation function for Actor and Critic networks
            print("\n=== Configuring activation function for Actor network ===", flush = True)
            activation_function_actor, activation_params_actor = self._get_activation_config('Actor')
            print("\n=== Configuring activation function for Critic network ===", flush = True)
            activation_function_critic, activation_params_critic = self._get_activation_config('Critic')

            print("\n=== FQE Configuration ===", flush = True)
            print("\nAvailable optimizer options:", flush = True)
            print("- torch.optim.SGD", flush = True)
            print("- torch.optim.Adam", flush = True)
            print("- torch.optim.AdamW", flush = True)
            print("- torch.optim.RMSprop", flush = True)
            print("- torch.optim.Adagrad", flush = True)

            optim_fqe = input("Enter the optimizer for FQE agents: ")
            weight_decay_fqe = self._get_weight_decay(optim_fqe)

            print("\nAvailable loss functions:", flush = True)
            print("- nn.MSELoss()", flush = True)
            print("- nn.L1Loss()", flush = True)
            print("- nn.SmoothL1Loss()", flush = True)
            print("- nn.CrossEntropyLoss()", flush = True)
            print("- nn.BCELoss()", flush = True)

            loss_fqe = input("Enter the loss function for FQE agents: ")

            # Configure activation function for FQE networks
            print("\n=== Configuring activation function for FQE networks ===", flush = True)
            activation_function_fqe, activation_params_fqe = self._get_activation_config('FQE')

            memory_capacity = self._get_int_input("Enter the memory capacity: ", 1)

            target_update = self._get_int_input("Enter the target update frequency: ", 1)
            tau = self._get_float_input("Enter the soft update parameter (tau): ", 0, 1)

            lr_fqe_obj = self._get_float_input("Enter the learning rate of FQE agent for evaluating the objective cost: ", 0)

            constraint_num = self._get_int_input("Enter the number of constraints: ", 0)
            lr_fqe_con_list = []
            lr_lambda_list = []
            threshold_list = []

            for i in range(constraint_num):
                lr_fqe_con = self._get_float_input(f"Enter the learning rate of FQE Agent for evaluating the constraint {i+1}: ", 0)
                lr_lambda = self._get_float_input(f"Enter the learning rate for dual variable $\lambda$ {i+1}: ", 0)
                threshold = self._get_float_input(f"Enter the value of constraint threshold {i+1}: ", 0)
                lr_fqe_con_list.append(lr_fqe_con)
                lr_lambda_list.append(lr_lambda)
                threshold_list.append(threshold)

            device_type = input("Enter the device type (cuda/cpu): ")

            # Configure Lagrange multiplier update method
            print("\n=== Configuring Lagrange multiplier update method ===", flush = True)
            print("Available lambda update methods:", flush = True)
            print("- None (default projected gradient without bound)", flush = True)
            print("- 'PG with bound' (projected gradient with bound)", flush = True)
            print("- 'EG with bound' (exponentiated gradient with bound)", flush = True)
            
            lambda_update_input = input("Enter the lambda update method (press Enter for None): ").strip()
            if lambda_update_input == "" or lambda_update_input.lower() == "none":
                lambda_update = None
                bound_lambda = None
            else:
                lambda_update = lambda_update_input
                if lambda_update in ['PG with bound', 'EG with bound']:
                    bound_lambda = self._get_float_input("Enter the bound for lambda (B): ", 0)
                else:
                    bound_lambda = None

            self.config = RLConfig_custom(algo_name, gamma, batch_size, 
                                          train_eps, train_eps_steps,
                                          # Actor and Critic configurations
                                          lr_actor, lr_critic,
                                          weight_decay_actor, weight_decay_critic,
                                          optim_actor, optim_critic, loss_critic,
                                          activation_function_actor, activation_params_actor,
                                          activation_function_critic, activation_params_critic,
                                          # FQE configurations
                                          weight_decay_fqe, 
                                          optim_fqe, 
                                          loss_fqe, 
                                          activation_function_fqe, activation_params_fqe,
                                          # General configurations
                                          memory_capacity, 
                                          target_update, tau, 
                                          lr_fqe_obj, lr_fqe_con_list, lr_lambda_list, 
                                          constraint_num, threshold_list, device_type,
                                          lambda_update, bound_lambda)

            return self.config
        except Exception as e:
            print(f"Error in configuration: {e}")
            return None

    def _get_float_input(self, prompt: str, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
        """
        Helper method to get a validated float input.
        
        Args:
            prompt (str): Input prompt to display
            min_value (Optional[float]): Minimum allowed value
            max_value (Optional[float]): Maximum allowed value
            
        Returns:
            float: Validated float input
        """
        while True:
            try:
                value = float(input(prompt))
                if (min_value is not None and value < min_value) or (max_value is not None and value > max_value):
                    raise ValueError(f"Value must be between {min_value} and {max_value}.")
                return value
            except ValueError as e:
                print(f"Invalid input: {e}")

    def _get_int_input(self, prompt: str, min_value: Optional[int] = None) -> int:
        """
        Helper method to get a validated integer input.
        
        Args:
            prompt (str): Input prompt to display
            min_value (Optional[int]): Minimum allowed value
            
        Returns:
            int: Validated integer input
        """
        while True:
            try:
                value = int(float(input(prompt)))
                if min_value is not None and value < min_value:
                    raise ValueError(f"Value must be at least {min_value}.")
                return value
            except ValueError as e:
                print(f"Invalid input: {e}")

    def _get_weight_decay(self, optimizer: str) -> float:
        """
        Helper method to get weight decay based on optimizer type.
        
        Args:
            optimizer (str): Optimizer name
            
        Returns:
            float: Weight decay value
        """
        if optimizer in ['torch.optim.SGD', 'torch.optim.AdamW']:
            return self._get_float_input(f"Enter the weight decay for {optimizer}: ", 0)
        else:
            return 0

    def _get_activation_config(self, network_type: str) -> Tuple[str, Dict[str, Any]]:
        """
        Configure activation function and its parameters for different network types.
        
        Args:
            network_type (str): The type of network ('Actor', 'Critic', or 'FQE') for which the activation function is being configured.
        
        Returns:
            Tuple[str, Dict[str, Any]]: Activation function name and its parameters
        """
        print(f"\nAvailable activation functions for {network_type} networks:")
        
        # Get supported activations based on network type
        if network_type == 'Actor':
            supported_activations = FCN_actor.get_supported_activations()
        elif network_type == 'Critic':
            supported_activations = FCN_critic.get_supported_activations()
        else:  # FQE
            supported_activations = FCN_fqe.get_supported_activations()

        for func_name, params_info in supported_activations.items():
            print(f"- {func_name}: {params_info}")

        activation_function = input(f"Enter the activation function for {network_type} networks: ")

        activation_params: Dict[str, Any] = {}

        # Configure parameters based on the chosen activation function
        if activation_function == 'leaky_relu':
            negative_slope = self._get_float_input("Enter negative_slope for LeakyReLU (default: 0.01): ", 0)
            if negative_slope != 0.01:  # Only add if different from default
                activation_params['negative_slope'] = negative_slope
                
        elif activation_function == 'elu':
            alpha = self._get_float_input("Enter alpha for ELU (default: 1.0): ", 0)
            if alpha != 1.0:  # Only add if different from default
                activation_params['alpha'] = alpha
                
        elif activation_function == 'softplus':
            use_custom = input("Do you want to customize Softplus parameters? (y/n): ").lower().strip()
            if use_custom in ['y', 'yes']:
                beta = self._get_float_input("Enter beta for Softplus (default: 1.0): ", 0)
                threshold = self._get_float_input("Enter threshold for Softplus (default: 20): ", 0)
                if beta != 1.0:
                    activation_params['beta'] = beta
                if threshold != 20:
                    activation_params['threshold'] = threshold
                    
        elif activation_function == 'prelu':
            num_params = self._get_int_input("Enter num_parameters for PReLU (default: 1): ", 1)
            if num_params != 1:  # Only add if different from default
                activation_params['num_parameters'] = num_params
        
        return activation_function, activation_params
    

class RLTraining:
    """
    Training class for Reinforcement Learning agents using Actor-Critic and FQE.
    
    This class handles the training process for Actor-Critic and FQE agents in a constrained offline RL setting.
    
    Args:
        cfg (RLConfig_custom): Configuration object containing all hyperparameters
        input_dim (int): Dimension of the state space
        output_dim (int): Selected action dimension
        train_data_loader (callable): Function that returns training batches
        val_data_loader (callable): Function that returns validation batches
    """
    
    def __init__(self, 
                 cfg: RLConfig_custom, 
                 input_dim: int, 
                 output_dim: int, 
                 train_data_loader: callable, 
                 val_data_loader: callable) -> None:
        """
        Initialize the training environment.
        
        Args:
            cfg (RLConfig_custom): Configuration object
            input_dim (int): State space dimension
            output_dim (int): Selected action dimension
            train_data_loader (callable): Training data loader function
            val_data_loader (callable): Validation data loader function
        """
        self.cfg = cfg
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader

        # Store for Actor models history
        self.actor_models_history: List[Dict[str, Any]] = []
        # Store for Critic models history
        self.critic_models_history: List[Dict[str, Any]] = []
        
        # Store for FQE models history
        self.fqe_obj_models_history: List[Dict[str, Any]] = []
        self.fqe_con_models_history: Dict[int, List[Dict[str, Any]]] = {}

    def actor_agent_config(self, critic_agent: Critic, hidden_layers: Optional[List[int]] = None, weight_decay: Optional[float] = None, seed: int = 1) -> Actor:
        """
        Configure and create an Actor agent.
        
        Args:
            critic_agent (Critic): The Critic agent to be used
            hidden_layers (Optional[List[int]]): Hidden layer sizes for the network
            weight_decay (Optional[float]): Weight decay value (uses config default if None)
            seed (int): Random seed for reproducibility
            
        Returns:
            Actor: Configured Actor agent
        """
        torch.manual_seed(seed)
        
        if weight_decay is None:
            weight_decay = self.cfg.weight_decay_actor
            
        agent_actor = Actor(
            cfg = self.cfg,
            input_dim = self.input_dim,
            output_dim = self.output_dim,
            hidden_layers = hidden_layers,
            weight_decay = weight_decay,
            critic_agent = critic_agent
        )
        return agent_actor
    
    def critic_agent_config(self, hidden_layers: Optional[List[int]] = None, weight_decay: Optional[float] = None, seed: int = 1) -> Critic:
        """
        Configure and create a Critic agent.
        
        Args:
            hidden_layers (Optional[List[int]]): Hidden layer sizes for the network
            weight_decay (Optional[float]): Weight decay value (uses config default if None)
            seed (int): Random seed for reproducibility
            
        Returns:
            Critic: Configured Critic agent
        """
        torch.manual_seed(seed)
        
        if weight_decay is None:
            weight_decay = self.cfg.weight_decay_critic
            
        agent_critic = Critic(
            cfg = self.cfg,
            input_dim = self.input_dim + self.output_dim,  # state_dim + action_dim for Q-network
            output_dim = 1,  # Q-value output
            hidden_layers = hidden_layers,
            weight_decay = weight_decay
        )
        return agent_critic

    def fqe_agent_config(self, eval_agent: Actor, hidden_layers: Optional[List[int]] = None, 
                         weight_decay: Optional[float] = None, eval_target: Union[str, int] = 'obj', seed: int = 1) -> FQE:
        """
        Configure and create an FQE agent.
        
        Args:
            eval_agent (Actor): The policy agent to be evaluated
            hidden_layers (Optional[List[int]]): Hidden layer sizes for the network
            weight_decay (Optional[float]): Weight decay value (uses config default if None)
            eval_target (Union[str, int]): Target to evaluate ('obj' for objective, int for constraint index)
            seed (int): Random seed for reproducibility
            
        Returns:
            FQE: Configured FQE agent
        """
        torch.manual_seed(seed)
        
        if weight_decay is None:
            weight_decay = self.cfg.weight_decay_fqe
            
        agent_fqe = FQE(
            cfg = self.cfg,
            input_dim = self.input_dim,
            output_dim = self.output_dim,
            hidden_layers = hidden_layers,
            weight_decay = weight_decay,
            eval_agent = eval_agent,
            eval_target = eval_target
        )
        return agent_fqe
    
    def projected_gradient_update(self, lambda_list, B):
        if B == None:
            return [max(0, x) for x in lambda_list]
        else:
            norm = math.sqrt(sum(x*x for x in lambda_list))
            if norm > B:
                return [B * x / norm for x in lambda_list]
            else:
                return lambda_list
            
    def exponentiated_gradient(self, lambda_list, constraint_violation_list, lr_list, B):
        d = len(constraint_violation_list)
        constraint_violation_list_aug = constraint_violation_list + [0.0]
        lr_list_aug = lr_list + [0.0]

        w = [
            lambda_list[i] * math.exp(-lr_list_aug[i] * constraint_violation_list_aug[i])
            for i in range(d + 1)
        ]

        total_w = sum(w)

        return [wi/total_w * B for wi in w]

    def train(self, 
              agent_actor: Actor, 
              agent_critic: Critic, 
              agent_fqe_obj: FQE, 
              agent_fqe_con_list: List[FQE], 
              constraint: Optional[bool] = None,
              save_num: int = 100,
              z_value: float = 1.96):
        """
        Train the Actor-Critic and FQE agents using offline data.
        
        Args:
            agent_actor (Actor): The Actor agent for policy learning
            agent_critic (Critic): The Critic agent for Q-value estimation
            agent_fqe_obj (FQE): The FQE agent for objective evaluation
            agent_fqe_con_list (List[FQE]): List of FQE agents for constraint evaluation
            constraint (Optional[bool]): Whether to use constraint optimization
            save_num (int): Number of models to save
            z_value (float): Z-score for confidence intervals in Q-value estimation
            
        Returns:
            Following training metrics:
                - Actor loss
                - Critic loss
                - FQE objective loss  
                - FQE constraint loss
                - Critic estimated values
                - FQE estimated objective costs
                - FQE estimated constraint costs
                - Lambda (dual variable) values
        """
        print('Start to train!')
        print(f'Algorithm:{self.cfg.algo}, Device:{self.cfg.device}')

        self.Actor_loss: List[float] = []

        self.Critic_loss: List[float] = []
        self.Critic_est_values: List[float] = []

        self.FQE_loss_obj: List[float] = []
        self.FQE_loss_con: Dict[int, List[float]] = {i: [] for i in range(len(agent_fqe_con_list))}

        self.FQE_est_obj_costs: List[float] = []
        self.FQE_est_con_costs: Dict[int, List[float]] = {i: [] for i in range(len(agent_fqe_con_list))}

        self.lambda_dict: Dict[int, List[float]] = {i: [] for i in range(len(agent_fqe_con_list))}
        
        # Initialize the model history dictionaries for constraint agents
        for i in range(len(agent_fqe_con_list)):
            self.fqe_con_models_history[i] = []

        # Initialize the lambda lists
        if self.cfg.lambda_update == 'EG with bound':
            lambda_t_list = [self.cfg.bound_lambda/(len(agent_fqe_con_list) + 1) for i in range(len(agent_fqe_con_list) + 1)]
        else:
            lambda_t_list = [0.0 for i in range(len(agent_fqe_con_list))]
            
        lambda_update_list = [0.0 for i in range(len(agent_fqe_con_list))]

        state_batch_val = self.val_data_loader(data_type = 'val')
        
        model_update_counter = 0  # Counter to track model updates

        for k in range(self.cfg.train_eps):
            loss_list_actor: List[float] = []
            loss_list_critic: List[float] = []
            loss_list_fqe_obj: List[float] = []
            loss_list_fqe_con: Dict[int, List[float]] = {i: [] for i in range(len(agent_fqe_con_list))}

            critic_est_list: List[float] = []
            fqe_est_obj: List[float] = []
            fqe_est_con: Dict[int, List[float]] = {i: [] for i in range(len(agent_fqe_con_list))}

            for j in tqdm(range(self.cfg.train_eps_steps)):

                state_batch, action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch = self.train_data_loader()
                state_action_batch = torch.cat((state_batch, action_batch), dim = 1)
                
                # update the Critic agent and Actor agent
                loss_critic = agent_critic.update(agent_actor, lambda_t_list, state_action_batch, obj_cost_batch, con_cost_batch, next_state_batch, done_batch)
                loss_actor = agent_actor.update(state_batch)

                loss_ev_obj = agent_fqe_obj.update(state_action_batch, obj_cost_batch, next_state_batch, done_batch)
                
                model_update_counter += 1

                # Save Critic model state
                if len(self.critic_models_history) >= save_num:
                    self.critic_models_history.pop(0)  
                self.critic_models_history.append({
                    'update_num': model_update_counter,
                    'epoch': k,
                    'step': j,
                    'model_state': self._get_model_state(agent_critic)
                })
                
                # Save Actor model state
                if len(self.actor_models_history) >= save_num:
                    self.actor_models_history.pop(0)  
                self.actor_models_history.append({
                    'update_num': model_update_counter,
                    'epoch': k,
                    'step': j,
                    'model_state': self._get_model_state(agent_actor)
                })

                # Save FQE objective model state
                if len(self.fqe_obj_models_history) >= save_num:
                    self.fqe_obj_models_history.pop(0) 
                self.fqe_obj_models_history.append({
                    'update_num': model_update_counter,
                    'epoch': k,
                    'step': j,
                    'model_state': self._get_model_state(agent_fqe_obj)
                })
                
                ##############################################################################################################
                loss_list_actor.append(loss_actor)
                loss_list_critic.append(loss_critic)
                loss_list_fqe_obj.append(loss_ev_obj)

                if constraint is None:
                    for m in range(len(agent_fqe_con_list)):
                        loss_con = agent_fqe_con_list[m].update(state_action_batch, con_cost_batch[m], next_state_batch, done_batch)
                        loss_list_fqe_con[m].append(loss_con)
                        
                        # Save FQE constraint model state
                        if len(self.fqe_con_models_history[m]) >= save_num:
                            self.fqe_con_models_history[m].pop(0)  
                        self.fqe_con_models_history[m].append({
                            'update_num': model_update_counter,
                            'epoch': k,
                            'step': j,
                            'model_state': self._get_model_state(agent_fqe_con_list[m])
                        })
                        
                        con_est_value, con_est_value_up, con_est_value_lb = agent_fqe_con_list[m].avg_Q_value_est(state_batch_val, z_value)
                        fqe_est_con[m].append(con_est_value)
                    
                    critic_est_value = agent_critic.avg_Q_value_est(state_batch_val)
                    avg_q_value_obj, avg_q_value_obj_up, avg_q_value_obj_lb = agent_fqe_obj.avg_Q_value_est(state_batch_val, z_value)

                    critic_est_list.append(critic_est_value)
                    fqe_est_obj.append(avg_q_value_obj)

                    lambda_update_list = [0.0 for i in range(len(agent_fqe_con_list))]
                    lambda_t_list = [0.0 for i in range(len(agent_fqe_con_list))]
                
                else:
                    for m in range(len(agent_fqe_con_list)):
                        loss_con = agent_fqe_con_list[m].update(state_action_batch, con_cost_batch[m], next_state_batch, done_batch)
                        loss_list_fqe_con[m].append(loss_con)
                        
                        # Save FQE constraint model state
                        if len(self.fqe_con_models_history[m]) >= save_num:
                            self.fqe_con_models_history[m].pop(0)
                            
                        self.fqe_con_models_history[m].append({
                            'update_num': model_update_counter,
                            'epoch': k,
                            'step': j,
                            'model_state': self._get_model_state(agent_fqe_con_list[m])
                        })
                        
                        con_est_value, con_est_value_up, con_est_value_lb = agent_fqe_con_list[m].avg_Q_value_est(state_batch_val, z_value)
                        fqe_est_con[m].append(con_est_value)
                        
                        # contraint violation check
                        lambda_update_list[m] = con_est_value_up - self.cfg.constraint_limit[m]

                    if self.cfg.lambda_update == None:
                        for m in range(len(agent_fqe_con_list)):
                            lambda_t_list[m] = lambda_t_list[m] + (self.cfg.lr_lam[m] * lambda_update_list[m])
                        lambda_t_list = self.projected_gradient_update(lambda_t_list, B = None)
                    elif self.cfg.lambda_update == 'PG with bound':
                        for m in range(len(agent_fqe_con_list)):
                            lambda_t_list[m] = lambda_t_list[m] + (self.cfg.lr_lam[m] * lambda_update_list[m])
                            # lambda_t_list[m] = max(0, lambda_t_list[m])
                        lambda_t_list = self.projected_gradient_update(lambda_t_list, B = self.cfg.bound_lambda)
                    elif self.cfg.lambda_update == 'EG with bound':
                        lambda_t_list = self.exponentiated_gradient(lambda_t_list, 
                                                                    constraint_violation_list = lambda_update_list, 
                                                                    lr_list = self.cfg.lr_lam, 
                                                                    B = self.cfg.bound_lambda)
                    
                    critic_est_value = agent_critic.avg_Q_value_est(state_batch_val)
                    avg_q_value_obj, avg_q_value_obj_up, avg_q_value_obj_lb = agent_fqe_obj.avg_Q_value_est(state_batch_val, z_value)

                    critic_est_list.append(critic_est_value)
                    fqe_est_obj.append(avg_q_value_obj)
                ######################################################################################
                if j % self.cfg.target_update == 0:
                    ### update the target agent for learning agent (Critic)
                    for target_param, policy_param in zip(agent_critic.target_net.parameters(), agent_critic.policy_net.parameters()):
                        target_param.data.copy_(self.cfg.tau * policy_param.data + (1 - self.cfg.tau) * target_param.data)

                    ### update the target agent for learning agent (Actor)
                    for target_param, policy_param in zip(agent_actor.target_net.parameters(), agent_actor.policy_net.parameters()):
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

            self.Actor_loss.append(np.mean(loss_list_actor))
            self.Critic_loss.append(np.mean(loss_list_critic))
            self.FQE_loss_obj.append(np.mean(loss_list_fqe_obj))

            self.Critic_est_values.append(np.mean(critic_est_list))
            self.FQE_est_obj_costs.append(np.mean(fqe_est_obj))

        print("Complete Training!")

        return self.Actor_loss, self.Critic_loss, self.FQE_loss_obj, self.FQE_loss_con, self.Critic_est_values, self.FQE_est_obj_costs, self.FQE_est_con_costs, self.lambda_dict
    
    def _get_model_state(self, agent: Union[Actor, Critic, FQE]) -> Dict[str, Any]:
        """
        Extract the model state from an agent.
        
        Args:
            agent (Union[Actor, Critic, FQE]): The agent to extract state from
            
        Returns:
            Dict[str, Any]: Dictionary containing deep copies of policy_net and target_net states
        """
        return {
            'policy_net': copy.deepcopy(agent.policy_net.state_dict()),
            'target_net': copy.deepcopy(agent.target_net.state_dict())
        }
    
    def _save_models_to_disk(self, save_num: int = 100, save_path: str = 'saved_models') -> None:
        """
        Save the models to disk.
        
        This method saves the last 100 model states for each agent type to disk
        for later analysis or model recovery.
        """       
        # Create directory for saved models if it doesn't exist
        os.makedirs(f'{save_path}/fqe_obj', exist_ok = True)
        
        # Save objective FQE models
        for idx, model_data in enumerate(self.fqe_obj_models_history[-save_num:]):
            torch.save(
                model_data['model_state'], 
                f'{save_path}/fqe_obj/model_{model_data["update_num"]}.pt'
            )
        
        # Save constraint FQE models
        for con_idx in self.fqe_con_models_history.keys():
            os.makedirs(f'{save_path}/fqe_con_{con_idx}', exist_ok = True)
            
            for idx, model_data in enumerate(self.fqe_con_models_history[con_idx][-save_num:]):
                torch.save(
                    model_data['model_state'], 
                    f'{save_path}/fqe_con_{con_idx}/model_{model_data["update_num"]}.pt'
                )
        
        print(f"Models saved to disk in '{save_path}/' directory")
    
    def load_fqe_model(self, agent: Union[Actor, Critic, FQE], model_path: str) -> Union[Actor, Critic, FQE]:
        """
        Load a saved model into an agent.
        
        Args:
            agent (Union[Actor, Critic, FQE]): The agent to load the model into
            model_path (str): Path to the saved model state
        
        Returns:
            Union[Actor, Critic, FQE]: The agent with loaded model weights
        """
        model_state = torch.load(model_path)
        agent.policy_net.load_state_dict(model_state['policy_net'])
        agent.target_net.load_state_dict(model_state['target_net'])
        return agent
    
def save_ocrl_models_and_data(agent_actor,
                              agent_critic,
                              agent_fqe_obj,
                              agent_fqe_con_list: List,
                              ocrl_training,
                              constraint_names: List[str] = None,
                              model_save_path: str = "./saved_models",
                              data_save_path: str = "./saved_data",
                              approx_method: str = "neural_network",
                              version: str = "v0",
                              custom_filename_prefix: str = None,
                              save_date: bool = True,
                              create_dirs: bool = True):
    """
    Save trained OCRL models and training data with flexible path and naming options.
    
    Parameters:
    -----------
    agent_actor : trained Actor model object
    agent_critic : trained Critic model object
    agent_fqe_obj : trained FQE model for objective cost
    agent_fqe_con_list : List, list of trained FQE model for constraint costs
    ocrl_training : training process object containing training metrics
    constraint_names : List[str], names for constraints (default: None, uses "con_0", "con_1", etc.)
    model_save_path : str, path to save model files (default: "./saved_models")
    data_save_path : str, path to save training data files (default: "./saved_data")
    approx_method : str, approximation method name for file naming (default: "neural_network")
    version : str, model version number (default: "v0")
    custom_filename_prefix : str, custom prefix for filenames (default: None, uses "ocrl_agent")
    save_date : bool, whether to include date in filename (default: True)
    create_dirs : bool, whether to create directories if they don't exist (default: True)
    
    Returns:
    --------
    dict : Dictionary containing saved file paths for reference
    """
    
    # Generate date string if needed
    date_str = datetime.datetime.now().strftime("%Y%m%d") if save_date else ""
    
    # Set filename prefix
    if custom_filename_prefix is None:
        filename_prefix = "ocrl_agent"
    else:
        filename_prefix = custom_filename_prefix
    
    # Set constraint names if not provided
    if constraint_names is None:
        constraint_names = [f"con_{i}" for i in range(len(agent_fqe_con_list))]
    elif len(constraint_names) != len(agent_fqe_con_list):
        print(f"Warning: Number of constraint names ({len(constraint_names)}) doesn't match number of constraint agents ({len(agent_fqe_con_list)})")
        constraint_names = [f"con_{i}" for i in range(len(agent_fqe_con_list))]
    
    # Create filename components
    date_part = f"_{date_str}" if date_str else ""
    version_part = f"_{version}" if version else ""
    
    # Create directories if specified
    if create_dirs:
        os.makedirs(model_save_path, exist_ok = True)
        os.makedirs(data_save_path, exist_ok = True)
        print(f"Directories created/verified: {model_save_path}, {data_save_path}")
    
    # Dictionary to store saved file paths
    saved_files = {
        'models': {},
        'training_data': {}
    }
    
    try:
        # ========= Save models =========
        model_files = {
            'actor': f"{model_save_path}/{filename_prefix}_actor{date_part}{version_part}.pth",
            'critic': f"{model_save_path}/{filename_prefix}_critic{date_part}{version_part}.pth",
            'fqe_obj': f"{model_save_path}/{filename_prefix}_fqe_obj{date_part}{version_part}.pth"
        }
        
        # Add constraint models dynamically
        for i, constraint_name in enumerate(constraint_names):
            model_files[f'fqe_con_{constraint_name}'] = f"{model_save_path}/{filename_prefix}_fqe_con_{constraint_name}{date_part}{version_part}.pth"
        
        # Save main models
        torch.save(agent_actor, model_files['actor'])
        torch.save(agent_critic, model_files['critic'])
        torch.save(agent_fqe_obj, model_files['fqe_obj'])
        
        # Save constraint models
        for i, (constraint_name, agent_fqe_con) in enumerate(zip(constraint_names, agent_fqe_con_list)):
            torch.save(agent_fqe_con, model_files[f'fqe_con_{constraint_name}'])
        
        saved_files['models'] = model_files
        print(f"Models saved successfully to: {model_save_path}")
        print(f"  - Actor model: {model_files['actor']}")
        print(f"  - Critic model: {model_files['critic']}")
        print(f"  - FQE objective model: {model_files['fqe_obj']}")
        for constraint_name in constraint_names:
            print(f"  - FQE constraint ({constraint_name}) model: {model_files[f'fqe_con_{constraint_name}']}")
        
        # ========= Save training data =========
        data_files = {
            'actor_loss': f"{data_save_path}/{approx_method}_actor_loss{date_part}{version_part}.npy",
            'critic_loss': f"{data_save_path}/{approx_method}_critic_loss{date_part}{version_part}.npy",
            'critic_est_value': f"{data_save_path}/{approx_method}_critic_est_value{date_part}{version_part}.npy",
            'fqe_obj_loss': f"{data_save_path}/{approx_method}_fqe_obj_loss{date_part}{version_part}.npy",
            'fqe_est_obj': f"{data_save_path}/{approx_method}_fqe_est_obj{date_part}{version_part}.npy"
        }
        
        # Add constraint data files dynamically
        for i, constraint_name in enumerate(constraint_names):
            data_files[f'fqe_con_{constraint_name}_loss'] = f"{data_save_path}/{approx_method}_fqe_con_{constraint_name}_loss{date_part}{version_part}.npy"
            data_files[f'fqe_est_con_{constraint_name}'] = f"{data_save_path}/{approx_method}_fqe_est_con_{constraint_name}{date_part}{version_part}.npy"
            data_files[f'lambda_{constraint_name}'] = f"{data_save_path}/{approx_method}_lambda_{constraint_name}{date_part}{version_part}.npy"
        
        # Save basic training metrics
        np.save(data_files['actor_loss'], np.array(ocrl_training.Actor_loss))
        np.save(data_files['critic_loss'], np.array(ocrl_training.Critic_loss))
        np.save(data_files['critic_est_value'], np.array(ocrl_training.Critic_est_values))
        np.save(data_files['fqe_obj_loss'], np.array(ocrl_training.FQE_loss_obj))
        np.save(data_files['fqe_est_obj'], np.array(ocrl_training.FQE_est_obj_costs))
        
        # Save constraint-specific training metrics
        for i, constraint_name in enumerate(constraint_names):
            if i < len(ocrl_training.FQE_loss_con):
                np.save(data_files[f'fqe_con_{constraint_name}_loss'], np.array(ocrl_training.FQE_loss_con[i]))
            if i < len(ocrl_training.FQE_est_con_costs):
                np.save(data_files[f'fqe_est_con_{constraint_name}'], np.array(ocrl_training.FQE_est_con_costs[i]))
            if i < len(ocrl_training.lambda_dict):
                np.save(data_files[f'lambda_{constraint_name}'], np.array(ocrl_training.lambda_dict[i]))
        
        saved_files['training_data'] = data_files
        print(f"Training data saved successfully to: {data_save_path}")
        
        # Summary
        print(f"\n=== Save Summary ===")
        print(f"Date: {date_str if date_str else 'No date'}")
        print(f"Approximation method: {approx_method}")
        print(f"Version: {version}")
        print(f"Model save path: {model_save_path}")
        print(f"Data save path: {data_save_path}")
        print(f"Number of constraints: {len(constraint_names)}")
        print(f"Constraint names: {constraint_names}")
        print(f"Total files saved: {len(model_files) + len(data_files)}")
        print(f"===================")
        
        return saved_files
        
    except Exception as e:
        print(f"Error saving models and data: {e}")
        return None