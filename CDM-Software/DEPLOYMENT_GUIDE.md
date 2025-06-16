# Medical Decision Support System - Deployment Guide

## 🏥 System Overview
This system is a reinforcement learning-based critical care assessment system, designed specifically for ICU physicians to assist in deciding whether a patient is suitable for extubation or discharge.

## 📋 Pre-deployment Preparation

### System Requirements
- **Operating System**: Linux/Windows/MacOS
- **Memory**: At least 4GB RAM
- **Storage**: At least 10GB available space
- **Network**: Stable internet connection

### Required Software
1. **Docker** (recommended version ≥ 20.10)
2. **Docker Compose** (recommended version ≥ 1.29)

## 🚀 Quick Deployment (Docker Method)

### Method 1: One-click Deployment
```bash
# 1. Enter application directory
cd CDM-Software

# 2. Run deployment script
chmod +x deploy.sh
./deploy.sh
```

### Method 2: Manual Deployment
```bash
# 1. Build image
docker-compose build

# 2. Start services
docker-compose up -d

# 3. Check status
docker-compose ps
```

## 🌐 Other Deployment Options

### Option 2: Cloud Server Deployment

#### AWS EC2
1. Create EC2 instance (recommended t3.medium or higher configuration)
2. Install Docker and Docker Compose
3. Upload project files
4. Run deployment script

#### Alibaba Cloud ECS
1. Create ECS instance
2. Configure security groups (open ports 80 and 443)
3. Install Docker environment
4. Deploy application

### Option 3: Local Server Deployment
```bash
# 1. Install Python environment
python -m venv medical_app_env
source medical_app_env/bin/activate  # Linux/Mac
# medical_app_env\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt
pip install gunicorn

# 3. Start application
gunicorn --bind 0.0.0.0:5000 --workers 4 web_application_test:app
```

## 🔒 Security Configuration

### 1. Enable HTTPS
```bash
# Generate SSL certificate (Let's Encrypt)
sudo apt install certbot
sudo certbot certonly --standalone -d your-domain.com
```

### 2. Configure Firewall
```bash
# Open necessary ports
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### 3. Set Environment Variables
```bash
# Create .env file
echo "SECRET_KEY=your-super-secret-key" > .env
echo "FLASK_ENV=production" >> .env
```

## 📊 Monitoring and Maintenance

### View Application Logs
```bash
# Real-time log viewing
docker-compose logs -f medical-app

# View error logs
docker-compose logs medical-app | grep ERROR
```

### Health Check
```bash
# Check application status
curl http://localhost/
docker-compose ps
```

### Data Backup
```bash
# Backup model files
docker cp medical-app:/app/Software_FQE_models ./backup/
```

## 🏥 Physician User Guide

### Access Methods
- **Intranet Access**: http://server-ip-address
- **Domain Access**: http://your-domain.com
- **HTTPS Access**: https://your-domain.com

### Usage Workflow
1. Open browser and access system address
2. Select AI model
3. Input patient's 30 physiological indicators
4. Click submit to get extubation risk assessment
5. Use results to assist clinical decision-making

## 🛠️ Troubleshooting

### Common Issues
1. **Port occupied**: Modify port mapping in docker-compose.yml
2. **Insufficient memory**: Increase server memory or optimize model loading
3. **Missing model files**: Ensure model file paths are correct

### Contact Support
- Technical support email: maotong.sun@tum.de
- Emergency contact phone: +49-264-18825

## 📈 Performance Optimization

### 1. Increase Worker Processes
```yaml
# docker-compose.yml
command: ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "8", "web_application_test:app"]
```

### 2. Enable Caching
```python
# Add caching to Flask application
from flask_caching import Cache
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
```

### 3. Load Balancing
Use Nginx or other load balancers to distribute requests to multiple application instances.

## 📝 Update Guide

### Update Application
```bash
# 1. Stop services
docker-compose down

# 2. Pull latest code
git pull origin main

# 3. Rebuild and start
docker-compose up --build -d
```

## ⚠️ Important Reminders
- This system is only for assisting medical decisions and cannot replace professional medical judgment
- Regularly backup important data and model files
- It is recommended to enable HTTPS and access control in production environment 