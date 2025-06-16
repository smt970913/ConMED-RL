import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class FCN_fqe(nn.Module):
    """FQE neural network architecture - inference only"""
    def __init__(self, state_dim, action_dim):
        super(FCN_fqe, self).__init__()
        self.fc1 = nn.Linear(state_dim, 500)
        self.fc2 = nn.Linear(500, action_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope = 0.1)
        x = self.fc2(x)
        return x

class FCN_fqi(nn.Module):
    """FQI neural network architecture - inference only"""
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

class LightweightFQE:
    """Lightweight FQE model - inference only"""
    def __init__(self, state_dim=37, action_dim=2, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = FCN_fqe(state_dim, action_dim).to(self.device)
        
        # Simplified policy function - selects action corresponding to minimum Q value
        self.eval_policy = FCN_fqi(state_dim, action_dim).to(self.device)
    
    def load_from_full_model(self, full_model_path):
        """Extract network weights from full model"""
        try:
            # Load full model
            full_model = torch.load(full_model_path, map_location=self.device, weights_only=False)
            
            # Extract FQE network weights
            if hasattr(full_model, 'policy_net'):
                self.policy_net.load_state_dict(full_model.policy_net.state_dict())
                print(f"✓ Successfully loaded FQE network weights")
            
            # If model has eval_agent, also extract its weights
            if hasattr(full_model, 'eval_agent') and hasattr(full_model.eval_agent, 'policy_net'):
                self.eval_policy.load_state_dict(full_model.eval_agent.policy_net.state_dict())
                print(f"✓ Successfully loaded evaluation policy network weights")
            
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def load_weights_only(self, policy_net_path, eval_policy_path=None):
        """Load network weight files directly"""
        try:
            # Load FQE network weights
            self.policy_net.load_state_dict(torch.load(policy_net_path, map_location=self.device))
            print(f"✓ Successfully loaded FQE network weights: {policy_net_path}")
            
            # If evaluation policy weights are provided, load them too
            if eval_policy_path:
                self.eval_policy.load_state_dict(torch.load(eval_policy_path, map_location=self.device))
                print(f"✓ Successfully loaded evaluation policy weights: {eval_policy_path}")
            
            return True
        except Exception as e:
            print(f"❌ Failed to load weights: {e}")
            return False
    
    def get_policy_action(self, state_batch):
        """Get policy action - using simplified policy"""
        with torch.no_grad():
            q_values = self.eval_policy(state_batch)
            policy_action_batch = q_values.min(1)[1].unsqueeze(1)
            return policy_action_batch
    
    def avg_Q_value_est(self, state_batch):
        """Q-value estimation - main inference method"""
        self.policy_net.eval()
        self.eval_policy.eval()
        
        with torch.no_grad():
            # Get policy action
            policy_action_batch = self.get_policy_action(state_batch)
            
            # Calculate Q values
            q_values = self.policy_net(state_batch).gather(dim=1, index=policy_action_batch).squeeze(1)

            q_mean = q_values.mean()
            q_std = q_values.std()
            n = q_values.shape[0]
        
            if n <= 1 or q_std == 0:
                return q_mean.item(), q_mean.item()
        
            z = 2.33  # 99% confidence interval
            q_upper_bound = q_mean + z * (q_std / math.sqrt(n))
        
            return q_mean.item(), q_upper_bound.item()

class LightweightFQI:
    """Lightweight FQI model - inference only"""
    def __init__(self, state_dim=37, action_dim=2, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = FCN_fqi(state_dim, action_dim).to(self.device)
    
    def load_from_full_model(self, full_model_path):
        """Extract network weights from full model"""
        try:
            full_model = torch.load(full_model_path, map_location=self.device, weights_only=False)
            
            if hasattr(full_model, 'policy_net'):
                self.policy_net.load_state_dict(full_model.policy_net.state_dict())
                print(f"✓ Successfully loaded FQI network weights")
                return True
            else:
                print(f"❌ policy_net not found in model")
                return False
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def load_weights_only(self, policy_net_path):
        """Load network weight files directly"""
        try:
            self.policy_net.load_state_dict(torch.load(policy_net_path, map_location=self.device))
            print(f"✓ Successfully loaded FQI network weights: {policy_net_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to load weights: {e}")
            return False
    
    def avg_Q_value_est(self, state_batch):
        """Q-value estimation - FQI version"""
        self.policy_net.eval()
        
        with torch.no_grad():
            q_values = self.policy_net(state_batch)
            avg_q_values = q_values.min(1)[0].unsqueeze(1).mean().item()
            return avg_q_values

# Factory function: create lightweight model based on model type
def create_lightweight_model(model_type='fqe', state_dim=37, action_dim=2, device=None):
    """
    Create lightweight model
    
    Args:
        model_type: 'fqe' or 'fqi'
        state_dim: state dimension
        action_dim: action dimension
        device: computing device
    
    Returns:
        lightweight model instance
    """
    if model_type.lower() == 'fqe':
        return LightweightFQE(state_dim, action_dim, device)
    elif model_type.lower() == 'fqi':
        return LightweightFQI(state_dim, action_dim, device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}") 