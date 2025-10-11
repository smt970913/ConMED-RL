# 🎯 CDM-Software Demo Guide

## Overview

This guide explains how to run the **demonstration version** of the Clinical Decision Making (CDM) Software. The current version is designed to showcase the functionality of our OCRL-based decision support system through a web interface.

**⚠️ Important Note**: This is a **demonstration prototype**. Clinical deployment in collaboration with healthcare professionals is planned for future work.

## 📋 System Requirements

### Minimum Requirements
- **Operating System**: Windows 10+, Linux (Ubuntu 18.04+), or macOS 10.15+
- **Python**: 3.10.14 (recommended) or 3.8+
- **Memory**: At least 4GB RAM
- **Storage**: At least 2GB available space

## 🚀 Quick Start - Running the Demo

### Step 1: Ensure Dependencies are Installed

Make sure you have completed the installation following the main `README.md`:

```bash
# If not already installed
pip install -r requirements.txt
```

### Step 2: Navigate to CDM-Software Directory

```bash
cd CDM-Software
```

### Step 3: Run the Demo Application

**Option A: Using the Run Scripts (Recommended)**

**Windows Users:**
```cmd
run_app.bat
```

**Linux/Mac Users:**
```bash
chmod +x run_app.sh
./run_app.sh
```

**Option B: Direct Python Execution**

```bash
python web_application_demo.py
```

### Step 4: Access the Web Interface

Once the application starts, open your web browser and navigate to:
- **URL**: http://localhost:5000

You should see the ConMED-RL Clinical Decision Support interface.

## 🖥️ Using the Demo Interface

The demo application provides:

1. **Model Selection**: Choose from three FQE models:
   - Model 1: Objective cost estimation (e.g., mortality risk)
   - Model 2: Constraint cost 1 estimation (e.g., readmission risk)
   - Model 3: Constraint cost 2 estimation (e.g., ICU length-of-stay)

2. **Patient Data Input**: Input physiological variables and clinical measurements through an interactive form

3. **Risk Assessment**: View real-time decision support predictions based on trained OCRL models

4. **Results Visualization**: Review risk assessments with confidence intervals

## 🔍 Troubleshooting

### Common Issues

**Issue: "Port 5000 is already in use"**
- Solution: Stop any other applications using port 5000, or modify the port in `web_application_demo.py`

**Issue: "Module not found" errors**
- Solution: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue: "Model files not found"**
- Solution: Verify that the `Software_FQE_models/` directory exists in the project root

**Issue: Python not found**
- Solution: Ensure Python is installed and added to your system PATH

## 📝 Demo Limitations

This demonstration version:
- ✅ Showcases the core functionality of OCRL-based decision support
- ✅ Demonstrates the web interface for clinical decision-making
- ✅ Uses pre-trained FQE models for risk assessment
- ❌ Is **not** validated for clinical use
- ❌ Is **not** deployed in a production healthcare environment
- ❌ Does **not** include security features required for clinical deployment

## 🔮 Future Development

Clinical deployment with healthcare professionals will include:
- Security hardening and HIPAA compliance
- Integration with Electronic Health Record (EHR) systems
- Clinical validation and regulatory approval
- Comprehensive user training and documentation
- Production-grade infrastructure and monitoring

For information about planned Docker deployment capabilities (currently under development), see `../Docker-Deployment/README.md`.

## 📞 Support

For questions or issues with the demo:
- Contact: maotong.sun@tum.de, jingui.xie@tum.de
- Check the main documentation: `../README.md`