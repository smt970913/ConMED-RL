# 🚀 ConMED-RL Docker Quick Start Guide

## 🎯 What Can You Do Now?

Yes! Now you can in Docker on another computer:
- ✅ **Directly call** all functions inside `Data` and `ConMedRL`
- ✅ **Use** the clinical decision support system in CDM-Software
- ✅ **Open and run** all Jupyter notebooks in Experiment Notebook
- ✅ **Access** the complete development and research environment

## 📋 Three Deployment Modes

### 1. 🔬 Research Environment (Recommended for Data Analysis)

**One-click startup:**
```bash
# Linux/Mac
cd Docker-Deployment
chmod +x scripts/build_research.sh
./scripts/build_research.sh

# Windows
cd Docker-Deployment
scripts\build_research.bat
```

**What you can use:**
- 🌟 **Jupyter Lab**: http://localhost:8888 (password: `conmed-rl-research`)
- 📊 **All data analysis tools**: matplotlib, seaborn, plotly
- 🧠 **Complete ConMedRL framework**: directly import in notebooks
- 📁 **All project files**: including Experiment Notebook
- 🔧 **Flask application**: http://localhost:5000 (optional)

### 2. 💻 Development Environment (For Full-Stack Development)

```bash
cd Docker-Deployment
docker-compose -f docker-compose.dev.yml up --build -d
```

**What you can use:**
- 🌟 **Jupyter Lab**: http://localhost:8888 (password: `conmed-rl-dev`)
- 🌐 **Flask Web application**: http://localhost:5000
- 💾 **Database**: PostgreSQL (localhost:5432)

### 3. 🚀 Production Environment (For Deployment)

```bash
cd Docker-Deployment
docker-compose -f docker-compose.prod.yml up --build -d
```

**What you can use:**
- 🌐 **Web application**: http://localhost
- 📊 **Monitoring**: Prometheus + Grafana

## 🎓 Practical Usage Examples

### Using ConMedRL in Jupyter

```python
# In Jupyter notebook
import sys
sys.path.append('/app')

# Import core modules
from ConMedRL.conmedrl import FQI, FQE
from ConMedRL.data_loader import DataLoader

# Import data processing modules
from Data.mimic_iv_icu_discharge.data_preprocess import preprocess_data

# Usage example
data_loader = DataLoader()
fqi_agent = FQI()
fqe_agent = FQE()

# Data preprocessing
processed_data = preprocess_data('/app/Data/raw_data.csv')
```

### Running Experiment Notebook

```python
# Open directly in Jupyter Lab
# /app/Experiment Notebook/Case_ICU_Discharge_Decision_Making.ipynb
# /app/Experiment Notebook/Case_ICU_Extubation_Decision_Making.ipynb
# /app/Experiment Notebook/Example_dataset_preprocess_MIMIC-IV.ipynb
```

### Using CDM-Software

```python
# Start clinical decision support system
from CDM_Software.web_application_demo import app
app.run(host='0.0.0.0', port=5000)

# Or access directly at http://localhost:5000
```

## 📁 File Structure in Docker

```
/app/
├── ConMedRL/                    # Core OCRL framework
│   ├── conmedrl.py             # Main algorithm implementation
│   ├── conmedrl_continuous.py  # Continuous action space
│   └── data_loader.py          # Data loader
├── Data/                        # Data processing modules
│   ├── mimic_iv_icu_discharge/
│   ├── mimic_iv_icu_extubation/
│   └── SICdb_*/
├── CDM-Software/                # Clinical decision support software
│   ├── web_application_demo.py
│   └── interactive_support.py
├── Experiment Notebook/         # Jupyter notebooks
│   ├── Case_ICU_Discharge_Decision_Making.ipynb
│   ├── Case_ICU_Extubation_Decision_Making.ipynb
│   └── Example_dataset_preprocess_MIMIC-IV.ipynb
└── Software_FQE_models/         # Trained models
    ├── discharge_decision_making/
    └── extubation_decision_making/
```

## 🔧 Common Commands

### Environment Management
```bash
# Start research environment
docker-compose -f docker-compose.research.yml up -d

# View logs
docker-compose -f docker-compose.research.yml logs -f

# Stop environment
docker-compose -f docker-compose.research.yml down

# Enter container
docker-compose -f docker-compose.research.yml exec conmed-rl-research bash
```

### Health Checks
```bash
# Check Jupyter Lab
curl -f http://localhost:8888/lab

# Check Flask application
curl -f http://localhost:5000/health

# Check container status
docker-compose -f docker-compose.research.yml ps
```

## 💡 Best Practices

### 1. Data Analysis Workflow
```bash
# 1. Start research environment
./scripts/build_research.sh

# 2. Open Jupyter Lab
# Visit http://localhost:8888

# 3. Create new notebook or open existing ones
# Import required modules and start analysis
```

### 2. Development Workflow
```bash
# 1. Start development environment
docker-compose -f docker-compose.dev.yml up -d

# 2. Use both Jupyter and Flask simultaneously
# Jupyter: http://localhost:8888
# Flask: http://localhost:5000

# 3. Real-time debugging and testing
```

### 3. Deployment Workflow
```bash
# 1. Complete development in research environment
# 2. Test in development environment
# 3. Deploy to production environment
docker-compose -f docker-compose.prod.yml up -d
```

## 🆘 Common Issues and Solutions

### Port Conflicts
```bash
# Check port usage
netstat -tulpn | grep :8888

# Modify port (in docker-compose file)
ports:
  - "8889:8888"  # Change to 8889
```

### Module Import Failures
```python
# Add path in Jupyter
import sys
sys.path.append('/app')

# Verify path
print(sys.path)

# Check if files exist
import os
os.listdir('/app/ConMedRL')
```

### Jupyter Access Issues
```bash
# Check token
docker-compose -f docker-compose.research.yml exec conmed-rl-research jupyter lab list

# Restart Jupyter
docker-compose -f docker-compose.research.yml restart
```

## 📞 Get Help

1. View complete documentation: `Docker-Deployment/README.md`
2. Run validation tests: `./scripts/test_deployment.sh`
3. View troubleshooting: `Docker-Deployment/DOCKER_VALIDATION_GUIDE.md`
4. Contact maintainers: maotong.sun@tum.de, jingui.xie@tum.de

## 🎉 Getting Started

Now you can:
1. Choose the appropriate environment mode
2. Run the corresponding startup script
3. Access Jupyter Lab in your browser
4. Begin your ConMED-RL research journey!

**Recommended for first-time use:**
```bash
cd Docker-Deployment
chmod +x scripts/build_research.sh
./scripts/build_research.sh
```

Then visit http://localhost:8888 to get started!
