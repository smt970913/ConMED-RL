# Docker Deployment Guide

This directory contains all Docker-related files for deploying ConCare-RL in different environments.

## Directory Structure

```
Docker-Deployment/
├── README.md                    # This file
├── Dockerfile                   # Main Dockerfile for the application
├── docker-compose.yml           # Docker Compose configuration
├── docker-compose.prod.yml      # Production Docker Compose configuration
├── .dockerignore               # Docker ignore file
├── nginx.conf                  # Nginx configuration for production
├── scripts/
│   ├── build.sh               # Build script (Linux/Mac)
│   ├── build.bat              # Build script (Windows)
│   ├── deploy.sh              # Deployment script (Linux/Mac)
│   ├── deploy.bat             # Deployment script (Windows)
│   └── cleanup.sh             # Cleanup script
└── env/
    ├── .env.example           # Environment variables example
    └── .env.production        # Production environment variables
```

## Quick Start

### Development Environment

1. **Build and run with Docker Compose:**
   ```bash
   cd Docker-Deployment
   docker-compose up --build
   ```

2. **Access the application:**
   - Web interface: http://localhost:5000
   - Health check: http://localhost:5000/health

### Production Environment

1. **Build for production:**
   ```bash
   cd Docker-Deployment
   docker-compose -f docker-compose.prod.yml up --build -d
   ```

2. **Access the application:**
   - Web interface: http://localhost (port 80)
   - HTTPS: https://localhost (port 443, if SSL configured)

### Using Scripts

#### Windows
```cmd
cd Docker-Deployment\scripts
build.bat
deploy.bat
```

#### Linux/Mac
```bash
cd Docker-Deployment/scripts
chmod +x *.sh
./build.sh
./deploy.sh
```

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_ENV` | Flask environment | `production` |
| `FLASK_APP` | Flask application entry point | `web_application_test.py` |
| `SECRET_KEY` | Flask secret key | `change-me-in-production` |
| `WORKERS` | Number of Gunicorn workers | `2` |
| `PORT` | Application port | `5000` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Health Checks

The application includes built-in health checks:
- **Endpoint:** `/health`
- **Interval:** 30 seconds
- **Timeout:** 10 seconds
- **Retries:** 3

## Deployment Scenarios

### 1. Local Development
- Use `docker-compose.yml`
- Hot reload enabled
- Debug mode on
- Exposed on port 5000

### 2. Production Deployment
- Use `docker-compose.prod.yml`
- Nginx reverse proxy
- SSL termination
- Health monitoring
- Log aggregation

### 3. Cloud Deployment
- Compatible with AWS ECS, Google Cloud Run, Azure Container Instances
- Includes health checks and proper logging
- Configurable scaling options

## Monitoring and Logging

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f medical-app

# Last 100 lines
docker-compose logs --tail=100 medical-app
```

### Health Monitoring
```bash
# Check container health
docker ps

# Check service status
docker-compose ps

# View health check logs
docker inspect --format='{{json .State.Health}}' container_name
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using the port
   netstat -tulpn | grep :5000
   
   # Stop conflicting services
   docker-compose down
   ```

2. **Permission Denied**
   ```bash
   # Fix script permissions (Linux/Mac)
   chmod +x scripts/*.sh
   ```

3. **Build Failures**
   ```bash
   # Clean Docker cache
   docker system prune -a
   
   # Rebuild without cache
   docker-compose build --no-cache
   ```

4. **Container Won't Start**
   ```bash
   # Check logs
   docker-compose logs medical-app
   
   # Check container status
   docker-compose ps
   ```

### Performance Tuning

1. **Adjust Worker Count**
   - Edit `WORKERS` environment variable
   - Rule of thumb: 2 × CPU cores + 1

2. **Memory Limits**
   - Set memory limits in docker-compose.yml
   - Monitor usage with `docker stats`

3. **Volume Optimization**
   - Use named volumes for better performance
   - Avoid bind mounts in production

## Security Considerations

1. **Environment Variables**
   - Never commit real secrets to version control
   - Use Docker secrets or external secret management
   - Rotate keys regularly

2. **Network Security**
   - Use custom Docker networks in production
   - Implement proper firewall rules
   - Enable HTTPS in production

3. **Container Security**
   - Run as non-root user (already configured)
   - Keep base images updated
   - Scan for vulnerabilities regularly

## Migration from Old Setup

If you're migrating from the old Docker setup:

1. **Stop old containers:**
   ```bash
   docker stop $(docker ps -q)
   docker rm $(docker ps -aq)
   ```

2. **Remove old images:**
   ```bash
   docker rmi conmedrl
   ```

3. **Use new deployment:**
   ```bash
   cd Docker-Deployment
   docker-compose up --build
   ```

## Support

For deployment issues:
- Check the logs first: `docker-compose logs -f`
- Review this documentation
- Contact maintainers: maotong.sun@tum.de, jingui.xie@tum.de 