# 🛠️ Local Development Guide

This guide explains how to run ConMED-RL applications locally without Docker.

## 🎯 When to Use Local Development

- **Quick Testing**: Testing changes without building Docker images
- **Development**: Active development and debugging
- **Learning**: Understanding the application structure
- **Limited Resources**: When Docker is not available

## 📋 Prerequisites

### System Requirements
- Python 3.8 or higher
- At least 4GB RAM
- 2GB free disk space

### Dependencies
All dependencies are listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Windows Users
```cmd
# Navigate to CDM-Software directory
cd CDM-Software

# Run the application
run_app.bat
```

### Linux/Mac Users
```bash
# Navigate to CDM-Software directory
cd CDM-Software

# Make script executable
chmod +x run_app.sh

# Run the application
./run_app.sh
```

## 🔧 What the Scripts Do

### Automatic Python Detection
The scripts automatically detect and use the available Python interpreter:
- `python`
- `py` (Windows)
- `python3`

### Dependency Checking
Before starting, the scripts verify that all required packages are installed:
- `flask`
- `torch`
- `numpy`
- `sklearn`
- `PIL`

### Virtual Environment Support
If a `venv` directory exists, the scripts automatically activate it.

### Error Handling
Clear error messages and suggestions for:
- Missing Python installation
- Missing dependencies
- Installation instructions

## 📖 Usage Examples

### Basic Usage
```bash
# Windows
run_app.bat

# Linux/Mac
./run_app.sh
```

### With Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
pip install -r requirements.txt
run_app.bat

# Linux/Mac
source venv/bin/activate
pip install -r requirements.txt
./run_app.sh
```

### Manual Run (Alternative)
```bash
# Direct Python execution
python web_application_demo.py

# Or with specific Python version
python3 web_application_demo.py
```

## 🌐 Access the Application

Once started, the application will be available at:
- **URL**: http://localhost:5000
- **Health Check**: http://localhost:5000/health

## 🛠️ Development Tips

### 1. Enable Debug Mode
Edit `web_application_demo.py` and change:
```python
app.run(host='0.0.0.0', port=5000, debug=True)
```

### 2. Use Different Port
```python
app.run(host='0.0.0.0', port=8080, debug=True)
```

### 3. Hot Reload
Debug mode enables automatic reloading when files change.

## 🔍 Troubleshooting

### Common Issues

#### 1. Python Not Found
```bash
# Check Python installation
python --version
# or
python3 --version

# Add Python to PATH (Windows)
# Follow installation guide at https://www.python.org/downloads/
```

#### 2. Missing Dependencies
```bash
# Install dependencies
pip install -r requirements.txt

# Or with specific Python version
python3 -m pip install -r requirements.txt
```

#### 3. Port Already in Use
```bash
# Check what's using port 5000
netstat -tulpn | grep :5000

# Kill process using port
# Linux/Mac: sudo kill -9 <PID>
# Windows: taskkill /F /PID <PID>
```

#### 4. Module Import Errors
```bash
# Check if you're in the right directory
cd CDM-Software

# Verify Python path
python -c "import sys; print(sys.path)"

# Install missing packages
pip install <package_name>
```

### 5. Permission Denied (Linux/Mac)
```bash
# Make script executable
chmod +x run_app.sh

# Check file permissions
ls -la run_app.sh
```

## 🔄 Switching Between Local and Docker

### From Local to Docker
```bash
# Stop local application (Ctrl+C)
cd ../Docker-Deployment
docker-compose -f docker-compose.research.yml up -d
```

### From Docker to Local
```bash
# Stop Docker containers
cd Docker-Deployment
docker-compose down

# Start local development
cd ../CDM-Software
./run_app.sh
```

## 📚 Additional Resources

- **Docker Guide**: `../Docker-Deployment/README.md`
- **Quick Start**: `../Docker-Deployment/QUICK_START_GUIDE.md`
- **Deployment**: `DEPLOYMENT_GUIDE.md`
- **Main Documentation**: `../README.md`

## 🎯 Best Practices

1. **Use virtual environments** for isolated development
2. **Keep dependencies updated** regularly
3. **Use Docker for production** deployment
4. **Test locally** before Docker deployment
5. **Monitor resource usage** during development

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify Python and dependency versions
3. Review error messages carefully
4. Contact maintainers: maotong.sun@tum.de, jingui.xie@tum.de 