# Docker Deployment Validation Guide

This guide is used to verify the Docker configuration of the ConMED-RL project on another computer.

## 📋 Prerequisites Check

### 1. Environment Requirements
- Docker Engine 20.10+
- Docker Compose 1.29+
- Git
- curl (for health checks)

### 2. Environment Verification
```bash
# Check Docker version
docker --version

# Check Docker Compose version
docker-compose --version

# Check Git
git --version

# Check curl
curl --version
```

## 🚀 Quick Validation Process

### Method 1: Using Automated Test Script (Recommended)

```bash
# Clone repository
git clone https://github.com/your-username/ICU-Decision-Making-OCRL.git
cd ICU-Decision-Making-OCRL/Docker-Deployment

# Linux/Mac
chmod +x scripts/test_deployment.sh
./scripts/test_deployment.sh

# Windows
scripts\test_deployment.bat
```

### Method 2: Manual Validation Steps

#### Step 1: Verify Configuration Files
```bash
cd ICU-Decision-Making-OCRL/Docker-Deployment

# Verify docker-compose syntax
docker-compose config

# Verify production environment configuration
docker-compose -f docker-compose.prod.yml config
```

#### Step 2: Test Development Environment
```bash
# Build and start development environment
docker-compose up --build -d

# Check container status
docker-compose ps

# Check health status
curl -f http://localhost:5000/health

# View logs
docker-compose logs conmed-rl-app

# Stop development environment
docker-compose down
```

#### Step 3: Test Production Environment
```bash
# Start production environment
docker-compose -f docker-compose.prod.yml up --build -d

# Check container status
docker-compose -f docker-compose.prod.yml ps

# Check health status
curl -f http://localhost/health

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Stop production environment
docker-compose -f docker-compose.prod.yml down
```

#### Step 4: Test Monitoring Stack (Optional)
```bash
# Start complete monitoring stack
docker-compose -f docker-compose.prod.yml --profile monitoring up -d

# Check monitoring services
curl -f http://localhost:9090    # Prometheus
curl -f http://localhost:3000    # Grafana

# Cleanup
docker-compose -f docker-compose.prod.yml --profile monitoring down
```

## ✅ Validation Checklist

### Configuration Validation
- [ ] `docker-compose.yml` syntax is correct
- [ ] `docker-compose.prod.yml` syntax is correct
- [ ] `Dockerfile` builds successfully
- [ ] All service names are consistent (conmed-rl-*)

### Functional Validation
- [ ] Development environment starts successfully
- [ ] Production environment starts successfully
- [ ] Health check endpoint responds normally
- [ ] Log output is normal
- [ ] Port mapping is correct

### Service Validation
- [ ] Main application service runs normally
- [ ] Nginx proxy works normally
- [ ] Monitoring services are accessible (if enabled)

## 🔧 Troubleshooting Guide

### Common Issues and Solutions

#### 1. Port Conflicts
```bash
# Error: port is already allocated
# Solution: Check port usage
netstat -tulpn | grep :5000
netstat -tulpn | grep :80

# Or modify port mapping
# Modify ports configuration in docker-compose.yml
```

#### 2. Image Build Failure
```bash
# Error: Build failed
# Solution: Clean Docker cache
docker system prune -a
docker-compose build --no-cache
```

#### 3. Container Startup Failure
```bash
# Check container logs
docker-compose logs conmed-rl-app

# Check container status
docker-compose ps

# Enter container for debugging
docker-compose exec conmed-rl-app /bin/bash
```

#### 4. Health Check Failure
```bash
# Check if application started normally
docker-compose logs conmed-rl-app

# Manually test endpoint
curl -v http://localhost:5000/health

# Check application dependencies
docker-compose exec conmed-rl-app python -c "import flask; print('Flask OK')"
```

#### 5. Network Connection Issues
```bash
# Check Docker network
docker network ls

# Check inter-service connections
docker-compose exec conmed-rl-app ping nginx
```

## 📊 Performance Validation

### Resource Usage Check
```bash
# Monitor container resource usage
docker stats

# Check memory usage
docker-compose exec conmed-rl-app free -h

# Check disk usage
docker system df
```

### Load Testing (Optional)
```bash
# Simple load test
for i in {1..100}; do curl -s http://localhost:5000/health; done

# Using ab tool
ab -n 100 -c 10 http://localhost:5000/health
```

## 🔐 Security Validation

### Security Configuration Check
- [ ] Running as non-root user
- [ ] Principle of least privilege
- [ ] No sensitive information in image
- [ ] Network isolation is correct

### Security Scanning (Optional)
```bash
# Scan image vulnerabilities
docker scan conmed-rl-app

# Check container permissions
docker inspect conmed-rl-app | grep -i user
```

## 📝 Validation Report Template

### Environment Information
- OS: `uname -a`
- Docker version: `docker --version`
- Docker Compose version: `docker-compose --version`

### Test Results
- [ ] Configuration validation passed
- [ ] Build successful
- [ ] Development environment deployment successful
- [ ] Production environment deployment successful
- [ ] Health checks passed
- [ ] Performance normal
- [ ] Security checks passed

### Issue Log
- Issues encountered and solutions
- Performance observations
- Improvement suggestions

## 📞 Support Information

If you encounter problems during validation:
1. View logs: `docker-compose logs -f`
2. Check GitHub Issues
3. Contact maintainers:
   - maotong.sun@tum.de
   - jingui.xie@tum.de

## 🎯 Best Practices

1. **Regular validation**: Perform validation after each update
2. **Environment consistency**: Ensure test environment matches production environment
3. **Documentation updates**: Update configuration documentation in a timely manner
4. **Backup strategy**: Regular backups of important data
5. **Monitoring and alerts**: Set up appropriate monitoring and alerting
