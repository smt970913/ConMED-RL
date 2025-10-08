# ConMED-RL: An OCRL-Based Toolkit for Medical Decision Support

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/conmedrl.svg)](https://badge.fury.io/py/conmedrl)

**ConMED-RL** is an **Offline Constrained Reinforcement Learning (OCRL)** toolkit designed for critical care decision support. The toolkit provides an OCRL-based policy learning framework for medical decision-making tasks under single or multiple constraints.

This toolkit builds upon our research on OCRL applications in critical care: a published study in *IISE Transactions on Healthcare Systems Engineering* addressing ICU discharge decision-making, and ongoing work under revision in *Health Care Management Science* on ICU extubation decision-making.

## 🚀 Quick Start

### Installation

Install ConMED-RL using pip:

```bash
pip install conmedrl
```

### Brief Usage Instruction
- Set Hyparameter
```python
from ConMedRL import RLConfigurator, RLConfig_custom 

dm_configuration = RLConfigurator()
dm_configuration.choose_config_method()
dm_configuration.config.memory_capacity
```

- Load data into the training procedure (discharge decision-making case)

```python
from ConMedRL import FQE, FQI, TrainDataLoader, ValTestDataLoader

# Load your clinical data
train_data_loader = TrainDataLoader(cfg = dm_configuration.config, 
                                    outcome_table = outcome_table_train, 
                                    state_var_table = state_var_table, 
                                    terminal_state = terminal_state)

train_data_loader.data_buffer_train(action_name = 'discharge_action', 
                                    done_condition = True, 
                                    num_constraint = 2)

val_data_loader = ValTestDataLoader(cfg = dm_configuration.config, 
                                    outcome_table_select = outcome_table_val_select, 
                                    state_var_table_select = state_var_table_val_select, 
                                    outcome_table = outcome_table_val, 
                                    state_var_table = state_var_table_val, 
                                    terminal_state = terminal_state)

val_data_loader.data_buffer(action_name = 'discharge_action', 
                            done_condition = True, 
                            num_constraint = 2)
```

- Initialize the training of OCRL-based policy learning framework (discharge decision-making case)
```python
# Set the training of the model
ocrl_training = RLTraining(cfg = dm_configuration.config, 
                           state_dim = state_var_table.shape[1], 
                           action_dim = 2, 
                           train_data_loader = train_data_loader.data_torch_loader_train,
                           val_data_loader = val_data_loader.data_torch_loader)

# Building the FQI agent
fqi_agent = ocrl_training.fqi_agent_config(...) 

# Building the FQE agents
fqe_agent_obj = ocrl_training.fqe_agent_config(...) 

fqe_agent_con_0 = ocrl_training.fqe_agent_config(...) 

fqe_agent_con_1 = ocrl_training.fqe_agent_config(...) 

ocrl_training.train(agent_fqi = fqi_agent, 
                    agent_fqe_obj = fqe_agent_obj, 
                    agent_fqe_con_list = [fqe_agent_con_0, fqe_agent_con_1], 
                    constraint = True,
                    save_num = 100,
                    z_value = 1.96)
```

## 📦 Core Components

### Offline Constrained Reinforcement Learning (OCRL) Algorithms

- **Fitted Q-Evaluation (FQE)**: Policy evaluation for offline data
- **Fitted Q-Iteration (FQI)**: Policy optimization 
- **Replay Buffer**: Efficient data management for training
- **Custom RL Configurator**: Flexible configuration for different clinical scenarios

### Data Processing

- **TrainDataLoader**: Handles training data preparation and batch generation
- **ValTestDataLoader**: Manages validation and testing data processing
- Support for custom done conditions and constraint cost functions

## 🏥 Key Features

- **Offline Learning**: Train models on historical clinical data without online interaction
- **Constraint Handling**: Built-in support for clinical safety/efficiency constraints
- **Flexible Architecture**: Easy integration with existing clinical datasets
- **Medical Focus**: Specifically designed for critical care decision-making scenarios
- **Research-Backed**: Based on peer-reviewed methodologies

## 📊 Use Cases

ConMED-RL has been successfully applied to:

- **ICU Discharge Decision-Making**: Optimizing timing and safety of patient discharge
- **ICU Mechanical Ventilation Weaning**: Supporting extubation decisions with constraint satisfaction
- **Multi-Constraint Clinical Decisions**: Balancing multiple clinical objectives requirements

## 🔧 Hyperparameter Configuration

```python
from ConMedRL import RLConfig_custom

custom_config = RLConfig_custom(
    algo_name = 'OCRL policy learning for ...'
    gamma = 0.99,
    batch_size = 256,
    train_eps = int(8e6),
    ...
)

# Use custom configuration in training
trainer = RLTraining(
    config = custom_config, 
    ...
)
```

## Data Preprocessing

ConMED-RL expects data in MDP format suitable for offline RL training:
- **State Table**: Physiological measurements and clinical variables
- **Outcome Table**: Actions, costs/rewards, and terminal indicators

See the [full documentation](https://github.com/smt970913/ConMED-RL) for data preprocessing examples.

## 📖 Documentation and Examples

For comprehensive guides, tutorials, and examples:

- **GitHub Repository**: [https://github.com/smt970913/ConMED-RL](https://github.com/smt970913/ConMED-RL)
- **Example Notebooks**: Interactive Jupyter notebooks for MIMIC-IV datasets
- **Web Application Demo**: Clinical decision support interface

## 🔬 Research and Citation

This toolkit is based on research published in academic journals. If you use ConMED-RL in your research, please cite:

```bibtex
@misc{sun2025comedRL,
  author       = {Maotong Sun and Jingui Xie},
  title        = {ConMED-RL: An OCRL-Based Toolkit for Medical Decision Support},
  year         = {2025},
  howpublished = {\url{https://github.com/smt970913/ConMED-RL}},
  note         = {Accessed: xxxx-xx-xx},
}
```

## 🛠️ Requirements

- Python 3.8 or higher
- PyTorch
- NumPy
- Pandas
- scikit-learn

All dependencies are automatically installed with the package.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/smt970913/ConMED-RL/blob/main/LICENSE) file for details.

## 👥 Authors and Contact

- **Maotong Sun** - maotong.sun@tum.de
- **Jingui Xie** - jingui.xie@tum.de

School of Management, Technical University of Munich

## 🤝 Contributing

We welcome contributions! For major changes, please open an issue first to discuss what you would like to change.

For development setup and contributing guidelines, visit the [GitHub repository](https://github.com/smt970913/ConMED-RL).

## 🔗 Links

- **PyPI**: [https://pypi.org/project/conmedrl/](https://pypi.org/project/conmedrl/)
- **GitHub**: [https://github.com/smt970913/ConMED-RL](https://github.com/smt970913/ConMED-RL)
- **Issues**: [https://github.com/smt970913/ConMED-RL/issues](https://github.com/smt970913/ConMED-RL/issues)
- **Documentation**: [https://github.com/smt970913/ConMED-RL#readme](https://github.com/smt970913/ConMED-RL#readme)

## ⚠️ Disclaimer

This toolkit is intended for research purposes. Clinical deployment requires appropriate validation, regulatory approval, and should only be used by qualified healthcare professionals in accordance with institutional guidelines and applicable regulations.

---

**Keywords**: reinforcement learning, constrained reinforcement learning, offline reinforcement learning, clinical decision support, healthcare, ICU, critical care, machine learning, artificial intelligence
