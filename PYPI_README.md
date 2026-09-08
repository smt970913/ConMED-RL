# ConMED-RL: Offline Constrained RL for Critical-Care Research

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/conmedrl.svg)](https://pypi.org/project/conmedrl/)

**ConMED-RL** is an **Offline Constrained Reinforcement Learning (OCRL)**
toolkit for retrospective critical-care research. It combines ICU data
processing, discrete and continuous constrained policy learning, separate
Fitted Q Evaluation (FQE) models for objectives and constraints, and
research-interface examples.

This toolkit builds upon our research on OCRL applications in critical care: a published study in *IISE Transactions on Healthcare Systems Engineering* addressing ICU discharge decision-making, and ongoing work under revision in *Health Care Management Science* on ICU extubation decision-making.

Current release: **1.1.0**.

## 🚀 Quick Start

### Installation

Install ConMED-RL using pip:

```bash
pip install conmedrl

# Optional d3rlpy/Parquet or LLM integrations
pip install "conmedrl[data]"
pip install "conmedrl[llm]"
pip install "conmedrl[data,llm]"
```

### Basic Usage

```python
from ConMedRL import (
    RLConfigurator,
    RLTraining,
    TrainDataLoader,
    ValTestDataLoader,
    build_dataset,
)

configuration = RLConfigurator()
configuration.choose_config_method()
rl_config = configuration.config

bundle = build_dataset(
    database="mimic-iv",
    task="discharge",
    data_dir="/path/to/mimic-iv",
    output_dir="./processed",
    output_formats=("csv",),
)

train_loader = TrainDataLoader(cfg=rl_config, **bundle.loader_kwargs("train"))
train_loader.data_buffer_train(
    action_name=bundle.loader_action,
    done_condition=None,
    num_constraint=bundle.num_constraints,
)

val_loader = ValTestDataLoader(cfg=rl_config, **bundle.loader_kwargs("val"))
val_loader.data_buffer(
    action_name=bundle.loader_action,
    done_condition=None,
    num_constraint=bundle.num_constraints,
)

trainer = RLTraining(
    cfg=rl_config,
    state_dim=bundle.state_dim,
    action_dim=bundle.action_dim,
    train_data_loader=train_loader.data_torch_loader_train,
    val_data_loader=val_loader.data_torch_loader,
)
```

See the end-to-end notebook for FQI/FQE configuration, multiplier updates, and
held-out evaluation.

## 📦 Core Components

### Offline Constrained Reinforcement Learning (OCRL) Algorithms

- **Fitted Q-Evaluation (FQE)**: Policy evaluation method for offline data
- **Fitted Q-Iteration (FQI)**: Value-based offline RL algorithm 
- **Replay Buffer**: Efficient data management for training
- **Custom RL Configurator**: Flexible configuration for different clinical scenarios

### Data Processing

- **TrainDataLoader**: Handles training data preparation and batch generation
- **ValTestDataLoader**: Manages validation and testing data processing
- Support for custom done conditions and constraint cost functions
- Unified MIMIC-IV/SICdb preprocessing and an approved declarative adapter for
  MIMIC-like datasets such as NWICU
- Review-first, metadata-only LLM planning for custom discrete or continuous
  clinical OCRL tasks; generated code is never executed
- CSV, Parquet, d3rlpy, and de-identified FHIR R4 NDJSON exchange outputs
- Auditable patient withdrawal with dataset/model version tracking,
  stale-model rejection, and caller-controlled fresh retraining

See
[`DATA_PROCESSING.md`](https://github.com/smt970913/ConMED-RL/blob/main/DATA_PROCESSING.md)
and the generic-data, FHIR, and exact-unlearning example notebooks for the full
safety and interoperability contract.

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
from ConMedRL import RLConfigurator

configuration = RLConfigurator()
configuration.choose_config_method()
rl_config = configuration.config
```

For a non-interactive, fully specified configuration and a short training run,
see `Experiment Notebook/Example_ConMedRL_End_to_End_Workflow.ipynb`.

## Data Preprocessing

ConMED-RL expects data in MDP format suitable for offline RL training:
- **State Table**: Physiological measurements and clinical variables
- **Outcome Table**: Actions, costs/rewards, and terminal indicators

See the [full documentation](https://github.com/smt970913/ConMED-RL) for data preprocessing examples.

## 📖 Documentation and Examples

For comprehensive guides, tutorials, and examples:

- **GitHub Repository**: [https://github.com/smt970913/ConMED-RL](https://github.com/smt970913/ConMED-RL)
- **Example Notebooks**: Interactive Jupyter notebooks for MIMIC-IV datasets
- **Web Application Demo**: Research interface for model-integration checks

## 🔬 Research and Citation

This toolkit is based on research published in academic journals. If you use ConMED-RL in your research, please cite:

```bibtex
@misc{sun2025comedRL,
  author       = {Maotong Sun and Jingui Xie},
  title        = {ConMED-RL: An OCRL-Based Toolkit for Medical Decision Support},
  year         = {2025},
  howpublished = {\url{https://github.com/smt970913/ConMED-RL}},
  note         = {Version 1.1.0},
}
```

## 🛠️ Requirements

- Python 3.8 or higher
- PyTorch
- NumPy
- Pandas
- scikit-learn

Core dependencies are installed with the package. Optional data and LLM
integrations use the `data` and `llm` extras shown above.

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

This toolkit is intended for retrospective research and software evaluation.
It is not a medical device and must not be used to direct patient care without
appropriate local validation, governance, regulatory review, and clinician
oversight.

---

**Keywords**: reinforcement learning, constrained reinforcement learning, offline reinforcement learning, clinical decision support, healthcare, ICU, critical care, machine learning, artificial intelligence
