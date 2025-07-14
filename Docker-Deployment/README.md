# Docker Deployment Guide

This directory contains all Docker-related files for deploying ConMED-RL in different environments.

## Directory Structure

```
Docker-Deployment/
├── README.md                    # This file
├── Dockerfile                   # Main Dockerfile for the application
├── docker-compose.yml           # Basic Docker Compose configuration
├── docker-compose.dev.yml       # Development environment with Jupyter
├── docker-compose.research.yml  # Research environment (Jupyter Lab focused)
├── docker-compose.prod.yml      # Production environment with monitoring
├── .dockerignore               # Docker ignore file
├── nginx.conf                  # Nginx configuration for production
├── DOCKER_VALIDATION_GUIDE.md  # Complete validation guide
├── scripts/
│   ├── build.sh               # Build script (Linux/Mac)
│   ├── build.bat              # Build script (Windows)
│   ├── build_research.sh      # Research environment build script
│   ├── build_research.bat     # Research environment build script (Windows)
│   ├── test_deployment.sh     # Deployment test script
│   ├── test_deployment.bat    # Deployment test script (Windows)
│   └── cleanup.sh             # Cleanup script
└── env/
    ├── .env.example           # Environment variables example
    └── .env.production        # Production environment variables
```

## Quick Start

### Method 1: Research Environment (Jupyter Lab + All Components)

**Perfect for:** Data analysis, experimentation, notebook development

```bash
# Linux/Mac
cd Docker-Deployment
chmod +x scripts/build_research.sh
./scripts/build_research.sh

# Windows
cd Docker-Deployment
scripts\build_research.bat
```

**Access:**
- Jupyter Lab: http://localhost:8888 (token: `conmed-rl-research`)
- Flask App: http://localhost:5000 (optional)

### Method 2: Development Environment (Jupyter + Flask)

**Perfect for:** Full-stack development, testing both research and web components

```bash
cd Docker-Deployment
docker-compose -f docker-compose.dev.yml up --build -d
```

**Access:**
- Jupyter Lab: http://localhost:8888 (token: `conmed-rl-dev`)
- Flask App: http://localhost:5000

### Method 3: Production Environment

**Perfect for:** Production deployment of the web application

```bash
cd Docker-Deployment
docker-compose -f docker-compose.prod.yml up --build -d
```

**Access:**
- Web interface: http://localhost (port 80)
- HTTPS: https://localhost (port 443, if SSL configured)

## Environment Comparison

| Environment | Jupyter Lab | Flask App | Database | Monitoring | Use Case |
|-------------|-------------|-----------|----------|------------|----------|
| Research    | ✅ (Primary) | ✅ (Optional) | ✅ (Optional) | ❌ | Data analysis, experimentation |
| Development | ✅ | ✅ | ✅ | ❌ | Full-stack development |
| Production  | ❌ | ✅ (Primary) | ❌ | ✅ | Production deployment |

## Available Components in Docker

### ✅ Included Components
- **ConMedRL** - Core OCRL framework (`/app/ConMedRL/`)
- **Data** - Data processing modules (`/app/Data/`)
- **CDM-Software** - Clinical decision support (`/app/CDM-Software/`)
- **Experiment Notebook** - Jupyter notebooks (`/app/Experiment Notebook/`)
- **Software_FQE_models** - Trained models (`/app/Software_FQE_models/`)

### 🔧 Usage Examples

#### Using ConMedRL in Jupyter
```python
# In Jupyter notebook
import sys
sys.path.append('/app')

from ConMedRL.conmedrl import FQI, FQE
from ConMedRL.data_loader import DataLoader
from Data.mimic_iv_icu_discharge.data_preprocess import preprocess_data

# Your code here...
```

#### Using Data Processing
```python
# Load and preprocess data
from Data.mimic_iv_icu_discharge.data_preprocess import preprocess_data

# Process your data
processed_data = preprocess_data('/app/Data/raw_data.csv')
```

#### Running Flask Application
```python
# Start the clinical decision support system
from CDM_Software.web_application_demo import app

app.run(host='0.0.0.0', port=5000)
```

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_ENV` | Flask environment | `production` |
| `FLASK_APP` | Flask application entry point | `web_application_demo.py` |
| `JUPYTER_ENABLE_LAB` | Enable Jupyter Lab | `yes` |
| `JUPYTER_TOKEN` | Jupyter access token | `conmed-rl-research` |
| `PYTHONPATH` | Python path | `/app` |
| `WORKERS` | Number of Gunicorn workers | `2` |
| `PORT` | Application port | `5000` |

### Jupyter Configuration

The research environment includes:
- **Jupyter Lab** with full extension support
- **Matplotlib**, **Seaborn**, **Plotly** for visualization
- **All project directories** mounted and accessible
- **Persistent volume** for Jupyter settings

## Monitoring and Logging

### View Logs
```bash
# Research environment
docker-compose -f docker-compose.research.yml logs -f

# Development environment
docker-compose -f docker-compose.dev.yml logs -f

# Production environment
docker-compose -f docker-compose.prod.yml logs -f
```

### Health Monitoring
```bash
# Check research environment
curl -f http://localhost:8888/lab

# Check development environment
curl -f http://localhost:5000/health
curl -f http://localhost:8888/lab

# Check production environment
curl -f http://localhost/health
```

## Data Persistence

### Jupyter Data
- Jupyter settings: `jupyter_data` volume
- Notebook outputs: `notebook_outputs` volume
- All project files: Mounted from host

### Database Data
- PostgreSQL data: `postgres_data` volume
- Persistent across container restarts

## Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep :8888
   netstat -tulpn | grep :5000
   
   # Modify port mapping in docker-compose files
   ```

2. **Jupyter Not Accessible**
   ```bash
   # Check logs
   docker-compose -f docker-compose.research.yml logs conmed-rl-research
   
   # Check token
   docker-compose -f docker-compose.research.yml exec conmed-rl-research jupyter lab list
   ```

3. **Import Errors in Jupyter**
   ```python
   # In Jupyter notebook
   import sys
   sys.path.append('/app')
   print(sys.path)
   
   # Check if modules are available
   import os
   os.listdir('/app')
   ```

4. **Flask App Not Starting**
   ```bash
   # Check Flask logs
   docker-compose -f docker-compose.dev.yml logs conmed-rl-dev
   
   # Check Flask configuration
   docker-compose -f docker-compose.dev.yml exec conmed-rl-dev env | grep FLASK
   ```

## Security Considerations

1. **Jupyter Token Security**
   - Change default tokens in production
   - Use secure tokens for remote access
   - Consider IP restrictions

2. **Container Security**
   - Runs as non-root user
   - Minimal attack surface
   - Regular security updates

## Migration Guide

If you're upgrading from basic setup:

1. **Stop existing containers:**
   ```bash
   docker-compose down
   ```

2. **Choose new environment:**
   ```bash
   # For research
   docker-compose -f docker-compose.research.yml up -d
   
   # For development
   docker-compose -f docker-compose.dev.yml up -d
   ```

## Support

For deployment issues:
- Check logs: `docker-compose -f [config-file] logs -f`
- Review validation guide: `DOCKER_VALIDATION_GUIDE.md`
- Contact maintainers: maotong.sun@tum.de, jingui.xie@tum.de

## Best Practices

1. **Use research environment** for data analysis and experimentation
2. **Use development environment** for full-stack development
3. **Use production environment** for deployment
4. **Regular backups** of Jupyter notebooks and data
5. **Monitor resource usage** with `docker stats` 