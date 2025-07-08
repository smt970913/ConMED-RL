# ConMED-RL Library Usage Guide

This document provides a comprehensive guide on how to use ConMED-RL as a Python library for your own research and development projects.

## Installation

### From PyPI (Recommended)
```bash
pip install concarerl
```

### From Source
```bash
git clone https://github.com/your-username/ICU-Decision-Making-OCRL.git
cd ICU-Decision-Making-OCRL
pip install -e .
```

## Quick Start

### Basic Usage

```python
import ConMedRL
from ConMedRL import FQE, FQI, RLTraining, RLConfigurator
import numpy as np
import torch

# Create sample data
states = np.random.randn(1000, 10)  # 1000 samples, 10 features
actions = np.random.randint(0, 3, 1000)  # 3 possible actions
rewards = np.random.randn(1000)
next_states = np.random.randn(1000, 10)
dones = np.random.randint(0, 2, 1000)

# Configure the RL algorithm
config = RLConfigurator(
    state_dim=10,
    action_dim=3,
    hidden_dim=64,
    learning_rate=1e-3,
    batch_size=32
)

# Initialize FQE for policy evaluation
fqe = FQE(config)

# Train the FQE model
fqe.train(states, actions, rewards, next_states, dones, epochs=100)

# Evaluate a policy
policy_values = fqe.evaluate_policy(states, actions)
print(f"Average policy value: {np.mean(policy_values)}")
```

### Advanced Usage with Constraints

```python
from ConMedRL import FQI, ReplayBuffer

# Create replay buffer for experience storage
buffer = ReplayBuffer(capacity=10000, state_dim=10, action_dim=3)

# Add experiences to buffer
for i in range(1000):
    buffer.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

# Configure FQI with constraints
fqi_config = RLConfigurator(
    state_dim=10,
    action_dim=3,
    constraint_threshold=0.8,  # Safety constraint
    constraint_penalty=1.0,
    hidden_dim=128,
    learning_rate=1e-3
)

# Initialize FQI for constrained policy optimization
fqi = FQI(fqi_config)

# Train with safety constraints
fqi.train_with_constraints(
    buffer=buffer,
    constraint_function=lambda s, a: np.random.rand(),  # Example constraint
    epochs=200
)

# Get optimal actions for given states
optimal_actions = fqi.get_actions(states[:10])
print(f"Optimal actions: {optimal_actions}")
```

### Using Pre-trained Models

```python
from ConMedRL import RLTraining
import pickle

# Load a pre-trained model
with open('Software_FQE_models/discharge_decision_making/model.pkl', 'rb') as f:
    pretrained_model = pickle.load(f)

# Use the model for inference
test_states = np.random.randn(100, 10)
predictions = pretrained_model.predict(test_states)
print(f"Model predictions shape: {predictions.shape}")
```

### Data Processing

```python
from ConMedRL.data_loader import TrainDataLoader, ValTestDataLoader

# Initialize data loaders
train_loader = TrainDataLoader(
    data_path="path/to/your/training/data.csv",
    state_columns=['feature1', 'feature2', 'feature3'],
    action_column='action',
    reward_column='reward'
)

val_loader = ValTestDataLoader(
    data_path="path/to/your/validation/data.csv",
    state_columns=['feature1', 'feature2', 'feature3'],
    action_column='action',
    reward_column='reward'
)

# Load and preprocess data
train_data = train_loader.load_data()
val_data = val_loader.load_data()

print(f"Training data shape: {train_data['states'].shape}")
print(f"Validation data shape: {val_data['states'].shape}")
```

## Clinical Decision Making Examples

### ICU Extubation Decision Support

```python
from ConMedRL import FQE, FQI
import pandas as pd

# Load clinical data
clinical_data = pd.read_csv("your_icu_data.csv")

# Define clinical features
clinical_features = [
    'heart_rate', 'blood_pressure', 'oxygen_saturation',
    'respiratory_rate', 'temperature', 'consciousness_level'
]

# Prepare data for RL
states = clinical_data[clinical_features].values
actions = clinical_data['extubation_decision'].values  # 0: no extubation, 1: extubation
rewards = clinical_data['outcome_score'].values  # Clinical outcome metric

# Configure for clinical decision making
clinical_config = RLConfigurator(
    state_dim=len(clinical_features),
    action_dim=2,
    constraint_threshold=0.9,  # High safety requirement
    safety_constraint=True,
    hidden_dim=256,
    learning_rate=1e-4
)

# Train the model
clinical_fqi = FQI(clinical_config)
clinical_fqi.train_with_safety_constraints(
    states=states,
    actions=actions,
    rewards=rewards,
    safety_function=lambda s, a: clinical_safety_check(s, a),
    epochs=500
)

# Make clinical recommendations
patient_state = np.array([[75, 120, 98, 16, 37.2, 15]])  # Example patient
recommendation = clinical_fqi.get_actions(patient_state)
confidence = clinical_fqi.get_action_confidence(patient_state)

print(f"Recommendation: {'Extubate' if recommendation[0] == 1 else 'Do not extubate'}")
print(f"Confidence: {confidence[0]:.3f}")
```

### Discharge Decision Support

```python
# Similar approach for discharge decisions
discharge_features = [
    'length_of_stay', 'stability_score', 'complication_risk',
    'family_support', 'home_care_availability'
]

discharge_states = clinical_data[discharge_features].values
discharge_actions = clinical_data['discharge_decision'].values
discharge_rewards = clinical_data['readmission_penalty'].values

# Configure and train discharge model
discharge_fqi = FQI(clinical_config)
# ... training code similar to above
```

## Configuration Options

### RLConfigurator Parameters

```python
config = RLConfigurator(
    # Network architecture
    state_dim=10,                    # Dimension of state space
    action_dim=3,                    # Number of possible actions
    hidden_dim=128,                  # Hidden layer size
    num_layers=3,                    # Number of hidden layers
    
    # Training parameters
    learning_rate=1e-3,              # Learning rate
    batch_size=32,                   # Batch size for training
    discount_factor=0.99,            # Gamma for future rewards
    
    # Constraint parameters
    constraint_threshold=0.8,        # Safety constraint threshold
    constraint_penalty=1.0,          # Penalty for constraint violation
    safety_constraint=True,          # Enable safety constraints
    
    # Regularization
    l2_regularization=1e-4,          # L2 regularization strength
    dropout_rate=0.1,                # Dropout rate
    
    # Training options
    early_stopping=True,             # Enable early stopping
    patience=20,                     # Early stopping patience
    validation_split=0.2             # Validation data split
)
```

## Best Practices

### 1. Data Preprocessing
```python
# Normalize clinical features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
normalized_states = scaler.fit_transform(states)
```

### 2. Model Validation
```python
# Cross-validation for clinical models
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kfold.split(states):
    train_states, val_states = states[train_idx], states[val_idx]
    # Train and validate model
```

### 3. Safety Constraints
```python
# Define safety constraints for clinical decisions
def clinical_safety_check(state, action):
    """
    Safety check for clinical decisions
    Returns probability that action is safe given state
    """
    # Example: Check vital signs are within safe ranges
    heart_rate = state[0]
    if action == 1 and heart_rate > 120:  # High-risk action with high heart rate
        return 0.3  # Low safety probability
    return 0.9  # High safety probability
```

### 4. Model Interpretation
```python
# Get model explanations
feature_importance = fqi.get_feature_importance()
action_probabilities = fqi.get_action_probabilities(states)

# Visualize decision boundaries
import matplotlib.pyplot as plt
fqi.plot_decision_boundaries(feature_names=clinical_features)
plt.show()
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **CUDA Errors**: For GPU support, install PyTorch with CUDA
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Memory Issues**: Reduce batch size or use data generators
   ```python
   config.batch_size = 16  # Reduce batch size
   ```

4. **Convergence Issues**: Adjust learning rate and regularization
   ```python
   config.learning_rate = 1e-4  # Lower learning rate
   config.l2_regularization = 1e-3  # Stronger regularization
   ```

## Support

For questions and support:
- Check the [GitHub Issues](https://github.com/your-username/ICU-Decision-Making-OCRL/issues)
- Contact the maintainers: maotong.sun@tum.de, jingui.xie@tum.de
- Review the example notebooks in the `Experiment Notebook/` directory 