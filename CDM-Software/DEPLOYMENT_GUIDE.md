# 🚀 CDM-Software Deployment Guide

## 📋 System Requirements

### Minimum Requirements
- **Operating System**: Linux (Ubuntu 18.04+), Windows 10+, macOS 10.15+
- **Python**: 3.8 or higher
- **Memory**: At least 4GB RAM
- **Storage**: At least 10GB available space
- **Network**: Stable internet connection

### Required Software
1. **Docker** (recommended version ≥ 20.10)
2. **Docker Compose** (recommended version ≥ 1.29)

## 🚀 Quick Deployment (Docker Method - Recommended)

### ⚠️ Important: Docker Configuration Location
All Docker configurations have been moved to the `Docker-Deployment/` directory. 
**Do not use docker commands from this directory.**

### Method 1: Research Environment (Recommended)
```bash
# Navigate to Docker deployment directory
cd ../Docker-Deployment

# Use research environment (includes Jupyter Lab + Flask)
chmod +x scripts/build_research.sh
./scripts/build_research.sh
```

**Access:**
- **Jupyter Lab**: http://localhost:8888 (token: `conmed-rl-research`)
- **Flask App**: http://localhost:5000

### Method 2: Development Environment
```bash
# Navigate to Docker deployment directory
cd ../Docker-Deployment

# Start development environment
docker-compose -f docker-compose.dev.yml up --build -d
```

### Method 3: Production Environment
```bash
# Navigate to Docker deployment directory
cd ../Docker-Deployment

# Start production environment
docker-compose -f docker-compose.prod.yml up --build -d
```

## 📚 Complete Docker Documentation

For complete Docker setup instructions, please refer to:
- **Main Docker Guide**: `../Docker-Deployment/README.md`
- **Quick Start Guide**: `../Docker-Deployment/QUICK_START_GUIDE.md`
- **Validation Guide**: `../Docker-Deployment/DOCKER_VALIDATION_GUIDE.md`

## 🌐 Alternative Deployment Options

### Option 1: Local Python Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python web_application_demo.py
```

### Option 2: Cloud Server Deployment

#### AWS EC2
1. Create EC2 instance (recommended t3.medium or higher configuration)
2. Install Docker and Docker Compose
3. Clone the repository
4. Follow Docker deployment steps above

#### Google Cloud Platform
1. Create Compute Engine instance
2. Install Docker and Docker Compose
3. Clone the repository
4. Follow Docker deployment steps above

#### Azure Container Instance
1. Create Container Instance
2. Use the Docker images from Docker-Deployment
3. Configure environment variables