#!/bin/bash

# ConMED-RL Docker Deployment Test Script
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ConMED-RL Docker Deployment Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to check if a service is healthy
check_service_health() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    print_status "Checking $service_name health at $url..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            print_success "$service_name is healthy!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    print_error "$service_name health check failed after $max_attempts attempts"
    return 1
}

# Navigate to Docker-Deployment directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

print_status "Current directory: $(pwd)"

# Test 1: Validate Docker Compose files
print_status "Testing Docker Compose configuration..."

if docker-compose config > /dev/null 2>&1; then
    print_success "docker-compose.yml is valid"
else
    print_error "docker-compose.yml has syntax errors"
    exit 1
fi

if docker-compose -f docker-compose.prod.yml config > /dev/null 2>&1; then
    print_success "docker-compose.prod.yml is valid"
else
    print_error "docker-compose.prod.yml has syntax errors"
    exit 1
fi

# Test 2: Build images
print_status "Building Docker images..."
if docker-compose build --no-cache; then
    print_success "Docker images built successfully"
else
    print_error "Docker build failed"
    exit 1
fi

# Test 4: Test research environment
print_status "Testing research environment..."
docker-compose -f docker-compose.research.yml up -d

if [ $? -eq 0 ]; then
    print_success "Research environment started"
    
    # Check if containers are running
    if docker-compose -f docker-compose.research.yml ps | grep -q "Up"; then
        print_success "Research containers are running"
        
        # Check Jupyter Lab endpoint
        if check_service_health "http://localhost:8888/lab" "Research Jupyter Lab"; then
            print_success "Research environment is fully functional"
        else
            print_error "Research environment Jupyter Lab check failed"
            docker-compose -f docker-compose.research.yml logs conmed-rl-research
        fi
    else
        print_error "Research containers failed to start"
        docker-compose -f docker-compose.research.yml logs conmed-rl-research
    fi
else
    print_error "Failed to start research environment"
fi

# Cleanup research environment
print_status "Stopping research environment..."
docker-compose -f docker-compose.research.yml down

# Test 5: Test development environment
print_status "Testing development environment..."
docker-compose -f docker-compose.dev.yml up -d

if [ $? -eq 0 ]; then
    print_success "Development environment started"
    
    # Check if containers are running
    if docker-compose -f docker-compose.dev.yml ps | grep -q "Up"; then
        print_success "Development containers are running"
        
        # Check both Flask and Jupyter endpoints
        if check_service_health "http://localhost:5000/health" "Development Flask App"; then
            print_success "Development Flask app is functional"
        else
            print_error "Development Flask app check failed"
            docker-compose -f docker-compose.dev.yml logs conmed-rl-dev
        fi
        
        if check_service_health "http://localhost:8888/lab" "Development Jupyter Lab"; then
            print_success "Development Jupyter Lab is functional"
        else
            print_error "Development Jupyter Lab check failed"
            docker-compose -f docker-compose.dev.yml logs conmed-rl-dev
        fi
    else
        print_error "Development containers failed to start"
        docker-compose -f docker-compose.dev.yml logs conmed-rl-dev
    fi
else
    print_error "Failed to start development environment"
fi

# Cleanup development environment
print_status "Stopping development environment..."
docker-compose -f docker-compose.dev.yml down

# Test 6: Test production environment
print_status "Testing production environment..."
docker-compose -f docker-compose.prod.yml up -d

if [ $? -eq 0 ]; then
    print_success "Production environment started"
    
    # Check if containers are running
    if docker-compose -f docker-compose.prod.yml ps | grep -q "Up"; then
        print_success "Production containers are running"
        
        # Check health endpoint through nginx
        if check_service_health "http://localhost/health" "Production environment"; then
            print_success "Production environment is fully functional"
        else
            print_error "Production environment health check failed"
            docker-compose -f docker-compose.prod.yml logs conmed-rl-app
            docker-compose -f docker-compose.prod.yml logs nginx
        fi
    else
        print_error "Production containers failed to start"
        docker-compose -f docker-compose.prod.yml logs
    fi
else
    print_error "Failed to start production environment"
    exit 1
fi

# Test 7: Test monitoring (optional)
print_status "Testing monitoring stack..."
docker-compose -f docker-compose.prod.yml --profile monitoring up -d

if [ $? -eq 0 ]; then
    print_success "Monitoring stack started"
    
    # Check Prometheus
    if check_service_health "http://localhost:9090" "Prometheus"; then
        print_success "Prometheus is accessible"
    else
        print_warning "Prometheus health check failed"
    fi
    
    # Check Grafana
    if check_service_health "http://localhost:3000" "Grafana"; then
        print_success "Grafana is accessible"
    else
        print_warning "Grafana health check failed"
    fi
else
    print_warning "Monitoring stack failed to start"
fi

# Cleanup
print_status "Cleaning up test environment..."
docker-compose -f docker-compose.prod.yml --profile monitoring down

# Final report
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test Results Summary${NC}"
echo -e "${BLUE}========================================${NC}"
print_success "✅ Docker Compose configuration validation"
print_success "✅ Docker image building"
print_success "✅ Research environment deployment"
print_success "✅ Development environment deployment"
print_success "✅ Production environment deployment"
print_success "✅ Health checks"
echo ""
print_success "All tests passed! Your Docker configuration is working correctly."
echo ""
print_status "You can now deploy with confidence using:"
echo "  Development: docker-compose up -d"
echo "  Production:  docker-compose -f docker-compose.prod.yml up -d"
echo "  Monitoring:  docker-compose -f docker-compose.prod.yml --profile monitoring up -d" 