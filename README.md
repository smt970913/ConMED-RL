# ConCare-RL: An OCRL-Based Toolkit for Critical Decision Support

<div style="text-align: center;">
    <img src="image/ConCare-RL Logo.png" width="400">
</div>

This repository provides the implementation of an **Offline Constrained Reinforcement Learning (OCRL)** - based decision support toolkit for critical care decision-making. The toolkit is developed based on our research on **ICU extubation** and **discharge** decision-making. It builds on our studies in modeling and optimizing clinical decisions under uncertainty in the ICU setting.
In addition to the codebase, we provide two curated datasets designed to facilitate further research on extubation and discharge decision-making tasks.
This repository is created by **Maotong Sun** (maotong.sun@tum.de) and **Jingui Xie** (jingui.xie@tum.de).

## Repository Structure

```
ICU-Decision Making-OCRL/
│
├── CDM-Software/                            # Clinical Decision Making Software
│   ├── web_application_test.py              # Web application testing
│   ├── interactive_support.py               # Interactive decision support system
│   ├── test_environment.py                  # Environment testing utilities
│   ├── run_app.bat                          # Application runner (Windows)
│   └── DEPLOYMENT_GUIDE.md                  # Deployment documentation
│
├── ConCareRL/                               # Core OCRL framework
│   ├── __init__.py                          # Package initialization
│   ├── concarerl.py                         # Main OCRL implementation
│   ├── data_loader.py                       # DataLoader for sampling transitions
│   └── done_condition_function_examples.py  # Terminal condition examples
│
├── Docker-Deployment/                       # Docker deployment configurations
│   ├── README.md                            # Docker deployment guide
│   ├── Dockerfile                           # Optimized Docker image definition
│   ├── docker-compose.yml                   # Development environment setup
│   ├── docker-compose.prod.yml              # Production environment with monitoring
│   ├── nginx.conf                           # Nginx reverse proxy configuration
│   ├── .dockerignore                        # Docker ignore patterns
│   ├── scripts/
│   │   ├── build.sh                         # Build script (Linux/Mac)
│   │   ├── build.bat                        # Build script (Windows)
│   │   └── cleanup.sh                       # Docker cleanup script
│   └── env/
│       └── env.example                      # Environment variables template
│
├── Data/                                    # Datasets
│   ├── mimic_iv_icu_extubation/
│   │   └── data_preprocess.py               # MIMIC-IV extubation data preprocessing
│   ├── mimic_iv_icu_discharge/         
│   │   └── data_preprocess.py               # MIMIC-IV discharge data preprocessing
│   ├── SICdb_extubation/                    # Salzburg extubation dataset
│   └── SICdb_discharge/                     # Salzburg discharge dataset
│
├── Experiment Notebook/                     # Jupyter notebooks for experiments
│   ├── Example_MIMIC-IV_Extubation_Decision_Making.ipynb
│   ├── Example_MIMIC-IV_Discharge_Decision_Making.ipynb
│   ├── MIMIC_IV_dataset_prepare.ipynb
│   └── Salzburg_dataset_prepare.ipynb
│
├── Software_FQE_models/                     # Pre-trained models and software tools
│   ├── discharge_decision_making/           # Discharge decision models  
│   └── extubation_decision_making/          # Extubation decision models
│
├── image/                                   # Documentation images
│
├── README.md                                # Project documentation
├── LIBRARY_USAGE.md                         # Python library usage guide
├── requirements.txt                         # Python dependencies
├── setup.py                                 # Package setup file (setuptools)
├── pyproject.toml                           # Modern Python package configuration
├── build_package.py                         # Package build and publish script
├── MANIFEST.in                              # Package file inclusion rules
├── runtime.txt                              # Python runtime specification
└── Procfile                                 # Process file for deployment
```

## Installation Guideline

ConCare-RL supports multiple installation methods to accommodate different use cases and environments. Choose the method that best fits your needs:

### Prerequisites

- **Python 3.10.14** (recommended, as specified in `runtime.txt`)
- **Git** for cloning the repository
- **Docker** (optional, for containerized deployment)

### Method 1: Local Installation (Recommended for Development)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ICU-Decision-Making-OCRL.git
   cd ICU-Decision-Making-OCRL
   ```

2. **Create a virtual environment:**
   ```bash
   # Using conda (recommended)
   conda create -n concarerl python=3.10.14
   conda activate concarerl
   
   # OR using venv
   python -m venv concarerl_env
   # On Windows:
   concarerl_env\Scripts\activate
   # On Linux/Mac:
   source concarerl_env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the environment:**
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

5. **Verify installation:**
   ```bash
   python -c "import ConCareRL.concarerl; print('ConCare-RL installed successfully!')"
   ```

### Method 2: Docker Installation (Recommended for Production)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ICU-Decision-Making-OCRL.git
   cd ICU-Decision-Making-OCRL
   ```

2. **Quick start with automated scripts:**
   ```bash
   # On Windows:
   cd Docker-Deployment\scripts
   build.bat
   
   # On Linux/Mac:
   cd Docker-Deployment/scripts
   chmod +x build.sh
   ./build.sh
   ```

3. **Manual Docker commands:**
   ```bash
   # Navigate to Docker deployment directory
   cd Docker-Deployment
   
   # Development environment
   docker-compose up --build -d
   
   # Production environment
   docker-compose -f docker-compose.prod.yml up --build -d
   
   # With monitoring (Prometheus + Grafana)
   docker-compose -f docker-compose.prod.yml --profile monitoring up -d
   ```

4. **Access the application:**
   - **Development:** http://localhost:5000
   - **Production:** http://localhost (port 80)
   - **Monitoring:** http://localhost:3000 (Grafana), http://localhost:9090 (Prometheus)

### Method 3: PyPI Installation (Recommended for Core Framework)

For the core ConCare-RL framework without the web application:

```bash
# Install from PyPI (when published)
pip install concarerl

# Or install directly from GitHub
pip install git+https://github.com/your-username/ICU-Decision-Making-OCRL.git

# Install with optional dependencies
pip install concarerl[models,viz]  # For visualization and model utilities
pip install concarerl[dev]         # For development tools
```

### Quick Start for Different Use Cases

#### For Research and Experimentation:
1. Follow **Method 1** (Local Installation)
2. Launch Jupyter notebooks:
   ```bash
   jupyter notebook "Experiment Notebook/"
   ```
3. Start with example notebooks for MIMIC-IV datasets

#### For Clinical Decision Support:
1. Follow **Method 2** (Docker Installation)  
2. Use the web application:
   - Development: `http://localhost:5000`
   - Production: `http://localhost` (port 80)
3. Refer to `Docker-Deployment/README.md` for detailed deployment instructions

#### For Custom Development:
1. Follow **Method 1** (Local Installation)
2. Import ConCare-RL components:
   ```python
   from ConCareRL.concarerl import FQE, FQI, RLTraining
   from ConCareRL.data_loader import TrainDataLoader, ValTestDataLoader
   ```

### Dependencies Overview

The toolkit requires the following main dependencies:
- **PyTorch**: Deep learning framework for OCRL algorithms
- **Flask**: Web framework for the clinical decision support interface
- **scikit-learn**: Machine learning utilities
- **pandas/numpy**: Data manipulation and numerical computing
- **tqdm**: Progress bars for training processes

### Troubleshooting

**Common Installation Issues:**

1. **PyTorch installation issues:**
   ```bash
   # For CPU-only installation:
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   
   # For GPU support (CUDA):
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Memory issues with large datasets:**
   - Ensure sufficient RAM (minimum 8GB recommended)
   - Use data batching for large datasets

3. **Port conflicts (Docker):**
   ```bash
   # Change port mapping in docker-compose.yml or use:
   docker-compose up -d --scale web=1 -p 8080:5000
   ```

**Getting Help:**
- Check the `CDM-Software/DEPLOYMENT_GUIDE.md` for detailed deployment instructions
- Review example notebooks in `Experiment Notebook/` directory
- Contact the maintainers: maotong.sun@tum.de or jingui.xie@tum.de

## Implementation Guideline
<div style="text-align: center;">
    <img src="image/ConCareRL_pipeline.svg" width = 1400> 
</div>

### Dataset Module

The datasets are organized in the `Data/` folder with separate subdirectories for different clinical scenarios (discharge decision-making and extubation decision-making) and data sources (MIMIC-IV and SICdb).

#### Data Preprocessing Scripts
The toolkit includes specialized data preprocessing scripts located in `Data/*/data_preprocess.py` that transform diverse clinical datasets into standardized formats compatible with sequential decision-making frameworks. Our unified preprocessing architecture employs consistent Python classes and functions across all four `data_preprocess.py` implementations, with dataset-specific variations limited to the data selection phase. This design choice ensures reproducible data transformations while maintaining the flexibility needed to accommodate the unique characteristics of different medical datasets and research objectives.

**Key Feature**: Our preprocessing pipeline is implemented entirely in `Python` using `Dask` for efficient processing of large-scale clinical datasets (e.g., chartevents tables), **eliminating the need for SQL databases or complex data infrastructure**. This approach significantly simplifies the setup process and enhances reproducibility for researchers.

The preprocessing pipeline transforms raw clinical data into two structured pandas DataFrames that contain all necessary components for Markov Decision Process (MDP) formulation:
- **State Table**: Contains all physiological variables and clinical measurements that constitute the state space
- **Outcome Table**: Contains action decisions, terminal indicators (`done`), and reward information

These two tables provide the essential MDP components required for model-free offline reinforcement learning training.

#### Multi-language Support
For datasets that contain medical terminology in different languages (e.g., German in the SICdb), we provide an interface for calling large language model tools to perform translation and terminology lookup. **The large language model tools are used exclusively for translation purposes and are designed to maintain strict data privacy and confidentiality.**

#### Data Availability
The original medical research datasets (MIMIC-IV and SICdb) are not included in this repository due to privacy and licensing requirements. However, they can be obtained from authorized sources:

- **MIMIC-IV**: Available through [PhysioNet](https://physionet.org/content/mimiciv/3.1/) following proper data use agreements.
- **SICdb**: Available through [PhysioNet](https://physionet.org/content/sicdb/1.0.8/) following proper data use agreements.

Please ensure you have the appropriate permissions and follow all data use agreements when accessing these datasets.


### OCRL Algorithm Module
<div style="text-align: center;">
    <img src="image/dataset_usage.svg" width = 500> 
</div>

The OCRL Algorithm Module is the core component of ConCare-RL, implementing state-of-the-art offline constrained reinforcement learning algorithms specifically designed for critical care decision-making problems. This module bridges the gap between preprocessed clinical data and actionable decision support models.

#### Data Loading and Management

The `ConCareRL/data_loader.py` module transforms preprocessed pandas DataFrames into training-ready formats for offline reinforcement learning algorithms. This lightweight framework handles numerical data loading and batch preparation for model training. Alternatively, researchers can use the generated **Outcome Table** and **State Table** directly to construct MDP environments with established Python RL libraries such as `d3rlpy`.

**Core Components** - `data_loader.py`:

1. **`TrainDataLoader` Class**:
   - **Purpose**: Handles training data preparation and batch generation.
   - **Key Methods**:
     - `data_buffer_train()`: Loads training data into PyTorch sampling buffer.
     - `data_torch_loader_train()`: Returns PyTorch tensors for training.
   - **Features**:
     - Custom done condition function support.
     - Configurable constraint cost extraction.
     - Terminal state handling.
     - Memory buffer integration with `ReplayBuffer`.

2. **`ValTestDataLoader` Class**:
   - **Purpose**: Manages validation and testing data processing.
   - **Key Methods**:
     - `data_buffer()`: Loads validation/test data into buffer.
     - `data_torch_loader()`: Returns tensors for validation or testing.
   - **Features**:
     - Separate validation and test data handling.
     - Flexible data extraction modes.

**Terminal Condition Functions** - `done_condition_function_examples.py`:
- **`discharge_done_condition()`**: Example function for discharge decision-making scenarios.
- **Custom Logic**: Users can define domain-specific terminal conditions based on clinical outcomes.
- **Integration**: These functions are passed to data loader methods to determine episode termination.

**Data Flow Process**:
```
Preprocessed DataFrames → DataLoader → ReplayBuffer → PyTorch Tensors → RL Training
     ↓                        ↓              ↓               ↓               ↓
State/Outcome Tables → Buffer Loading → Memory Sampling → Batched Tensors → Model Updates
```

#### Core OCRL Framework

The `ConCareRL/concarerl.py` module implements the complete offline constrained reinforcement learning (OCRL) framework, featuring advanced algorithms specifically adapted for constrained medical decision-making.

**Approximation Model Architectures:**

1. **`FCN_fqe`**:
   - **Architecture**: Linear approximation/Fully connected neural network for Q-value estimation in policy evaluation.
   - **Features**: Configurable hidden layers, multiple activation functions, flexible architecture.

2. **`FCN_fqi`**:
   - **Architecture**: Linear approximation/Fully connected neural network for Q-value estimation in policy optimization.
   - **Features**: Similar architecture to FCN_fqe, optimized for constrained policy learning.

**Data Management:**

3. **`ReplayBuffer` Class**:
   - **Purpose**: Storage and sampling of offline clinical transitions.
   - **Key Methods**:
     - `push()`: Stores transitions (state, action, obj_cost, con_cost, next_state, done).
     - `sample()`: Random sampling for training batches.
     - `extract()`: Extract all stored data for validation/testing.

**Core Algorithms:**

4. **`FQE` (Fitted Q Evaluation) Class**:
   - **Purpose**: Evaluates the performance of existing clinical policies or policies derived by RL algorithm.
   - **Key Methods**:
     - `update()`: Updates Q-function using Bellman equation.
     - `avg_Q_value_est()`: Estimates average Q-values with confidence intervals.
     - `save()`: Saves trained model.
   - **Applications**: Policy evaluation and benchmarking.

5. **`FQI` (Fitted Q Iteration) Class**:
   - **Purpose**: Learns optimal decision policies with/without constraints.
   - **Key Methods**:
     - `update()`: Updates Q-function with constraint incorporation.
     - `avg_Q_value_est()`: Estimates average Q-values.
     - `rl_policy()`: Generates optimal actions given states.
     - `save()`: Saves trained model.
   - **Features**: Handles multiple constraints using Lagrange multipliers.

**Configuration and Training:**

6. **`RLConfig_custom` Class**:
   - **Purpose**: Configuration object containing all hyperparameters.
   - **Parameters**: Learning rates, batch size, network architectures, constraints, optimizers, loss functions.

7. **`RLConfigurator` Class**:
   - **Purpose**: Interactive configuration setup and management.
   - **Key Methods**:
     - `choose_config_method()`: Interactive configuration selection.
     - `load_config_from_json()`: Load configuration from file.
     - `save_config_to_json()`: Save configuration to file.
     - `input_rl_config()`: Manual configuration input.

8. **`RLTraining` Class**:
   - **Purpose**: Orchestrates the complete training pipeline.
   - **Key Methods**:
     - `fqi_agent_config()`: Initialize FQI agent.
     - `fqe_agent_config()`: Initialize FQE agent.
     - `train()`: Main training loop with constraint handling.
   - **Features**: Model saving, progress tracking, constraint satisfaction monitoring.

**Utility Functions:**
- **`save_ocrl_models_and_data()`**: Comprehensive model and data saving functionality.

For detailed usage examples and practical implementations, please refer to the **`Experiment Notebook/`** directory, which contains comprehensive Jupyter notebooks demonstrating:
- Data preprocessing and loading
- Model configuration and training
- Policy evaluation and optimization
- Clinical decision-making applications
   - Case 1: ICU Extubation decision-making
   - Case 2: ICU Discharge decision-making

This framework enables researchers and clinicians to develop, validate, and deploy offline constrained reinforcement learning systems for critical care environments.

### Software Module

The `CDM-Software/` directory contains the clinical decision support software implementation, providing an interactive web-based interface for healthcare professionals to utilize the trained OCRL models (the trained FQE models) in real clinical settings.

#### Interactive Decision Support System (`interactive_support.py`)

The core software module that implements the necessary Fitted Q Evaluation (FQE) model inside our OCRL framework optimized for real-time clinical decision support.

#### Web Application (`web_application_test.py`)

A `Flask`-based web application that provides an intuitive interface for clinical decision-making.

**Application Features:**

1. **Model Selection Interface**:
   - **Model 1**: FQE Estimated Objective Cost (primary outcome evaluation)
   - **Model 2**: FQE Estimated Constraint Cost 1 (e.g., readmission risk)
   - **Model 3**: FQE Estimated Constraint Cost 2 (e.g., ICU length-of-stay)

2. **Patient Data Input System**:
   - Interactive form for inputing the values of physiological variables
   - Real-time progress tracking and validation
   - Data Scaling with pre-stored scaler (if necessary)

3. **Clinical Decision Support**:
   - Real-time assessment using the trained FQE models
   - Visual feedback and confidence intervals

4. **Technical Implementation**:
   - `Flask` web framework with Bootstrap UI
   - `PyTorch` model integration
   - `Scikit-learn` preprocessing pipeline
   - Secure model loading and prediction

#### Testing and Validation Tools

**Environment Testing (`test_environment.py`)**:
- **Purpose**: Validates system dependencies and file availability
- **Features**:
  - Package import verification (`Flask`, `PyTorch`, `NumPy`, `Scikit-learn`, `PIL`)
  - Model file existence checking
  - Path validation for deployment
  - Comprehensive system readiness assessment

**Application Launcher (`run_app.bat`)**:
- **Purpose**: Cross-platform application startup script for Windows
- **Features**:
  - Multiple Python interpreter detection (python, py, python3)
  - Automatic error handling and user guidance
  - Installation instructions for missing dependencies

#### Deployment and Configuration

**Deployment Guide (`DEPLOYMENT_GUIDE.md`)**:
- **System Requirements**: Memory, storage, and network specifications
- **Deployment Methods**:
  - Docker containerization (recommended)
  - Cloud server deployment (AWS EC2, Alibaba Cloud ECS)
  - Local server installation
- **Security Configuration**: HTTPS, firewall, environment variables
- **Monitoring and Maintenance**: Logging, health checks, data backup
- **Clinical Usage Workflow**: Step-by-step physician user guide

**Key Deployment Features**:
- One-click Docker deployment
- Production-ready configuration with Gunicorn
- SSL/HTTPS support for secure clinical data handling
- Load balancing and performance optimization
- Comprehensive troubleshooting guide

#### Clinical Integration Workflow

```
Patient Data Input → Data Preprocessing → Model Selection → OCRL Prediction → Clinical Decision Support
       ↓                    ↓                  ↓               ↓                    ↓
37 Physiological → MinMax Scaling → FQE Model → Risk Assessment → Physician Review
   Parameters                                                                      ↓
                                                                          Clinical Decision
```

**Supported Clinical Scenarios**:
- **ICU Discharge Decision Support**: Risk assessment for patient discharge readiness
- **ICU Extubation Decision Support**: Evaluation of mechanical ventilation weaning for patients in ICU

This software module bridges the gap between research-grade OCRL algorithms and practical clinical deployment, providing healthcare professionals with AI-assisted decision support tools that maintain clinical safety standards while leveraging advanced Offline RL techniques.

For detailed deployment instructions and clinical usage guidelines, refer to the `CDM-Software/DEPLOYMENT_GUIDE.md` documentation.