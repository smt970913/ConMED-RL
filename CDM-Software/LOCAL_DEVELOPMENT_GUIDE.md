# 🛠️ Demo Application Setup Guide

This guide provides detailed instructions for running the ConMED-RL demo application locally.

## 🎯 Purpose

This guide is for users who want to:
- Run the demo application to see the interface and functionality
- Test the OCRL-based decision support system
- Understand how the web application works
- Experiment with the provided FQE models

**Note**: This is a **demonstration version** only. For information about future clinical deployment, see `DEPLOYMENT_GUIDE.md`.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.10.14 (recommended) or 3.8+
- **Memory**: At least 4GB RAM
- **Storage**: 2GB free disk space
- **Operating System**: Windows 10+, Linux (Ubuntu 18.04+), or macOS 10.15+

### Required Dependencies
All dependencies are listed in the main `requirements.txt` file. Install them with:
```bash
pip install -r requirements.txt
```

## 🚀 Running the Demo

### Method 1: Using Run Scripts (Recommended)

The run scripts automatically handle Python detection, dependency checking, and virtual environment activation.

**Windows:**
```cmd
cd CDM-Software
run_app.bat
```

**Linux/Mac:**
```bash
cd CDM-Software
chmod +x run_app.sh
./run_app.sh
```

### Method 2: Direct Python Execution

```bash
cd CDM-Software
python web_application_demo.py
```

### Method 3: With Virtual Environment

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install dependencies and run
pip install -r ../requirements.txt
python web_application_demo.py
```

## 🌐 Accessing the Demo

Once the application starts successfully, you will see:
```
 * Running on http://0.0.0.0:5000
 * Running on http://127.0.0.1:5000
```

Open your web browser and navigate to:
- **URL**: http://localhost:5000

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. Python Not Found
**Error**: `'python' is not recognized as an internal or external command`

**Solutions**:
- Verify Python installation: `python --version` or `python3 --version`
- Add Python to your system PATH
- Try using `python3` instead of `python`
- Download Python from https://www.python.org/downloads/

#### 2. Missing Dependencies
**Error**: `ModuleNotFoundError: No module named 'flask'` (or other packages)

**Solution**:
```bash
pip install -r requirements.txt
```

#### 3. Port Already in Use
**Error**: `Address already in use` or `Port 5000 is already in use`

**Solutions**:
- Stop the application using port 5000
- Windows: Find and kill the process using `netstat -ano | findstr :5000`
- Linux/Mac: Find and kill the process using `lsof -ti:5000 | xargs kill -9`

#### 4. Model Files Not Found
**Error**: `Model file not found` or path-related errors

**Solution**:
- Verify that `Software_FQE_models/` directory exists in the project root
- Ensure model files (.pth) are present in the appropriate subdirectories
- Check that you're running the script from the correct directory

#### 5. Permission Denied (Linux/Mac)
**Error**: `Permission denied: './run_app.sh'`

**Solution**:
```bash
chmod +x run_app.sh
```

## 🛠️ Optional: Debug Mode

For development and debugging, you can enable Flask's debug mode:

1. Open `web_application_demo.py`
2. Find the line: `app.run(host='0.0.0.0', port=5000)`
3. Change to: `app.run(host='0.0.0.0', port=5000, debug=True)`

Debug mode features:
- Automatic reload when code changes
- Detailed error messages in browser
- Interactive debugger

**⚠️ Warning**: Never use debug mode in production environments.

## 📚 Additional Resources

- **Demo Guide**: `DEPLOYMENT_GUIDE.md` - Overview and demo limitations
- **Main Documentation**: `../README.md` - Complete project documentation
- **Lightweight Version**: `LIGHTWEIGHT_MODEL_GUIDE.md` - Minimal inference-only version

## 💡 Tips

1. **Use Virtual Environments**: Keeps dependencies isolated and organized
2. **Check Console Output**: Watch for error messages and warnings
3. **Verify Model Files**: Ensure all required model files are present before starting
4. **Browser Compatibility**: Use modern browsers (Chrome, Firefox, Edge, Safari)

## 📞 Support

If you encounter issues not covered in this guide:
- Review error messages carefully for specific problems
- Check that all prerequisites are met
- Ensure you're using the correct Python version (3.10.14 recommended)
- Contact maintainers: maotong.sun@tum.de, jingui.xie@tum.de 