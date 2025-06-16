#!/bin/bash

# ConCare-RL Docker Build Script
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ConCare-RL Docker Build Script${NC}"
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

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed!"
    print_error "Please install Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not available!"
    print_error "Please install Docker Compose and try again."
    exit 1
fi

# Navigate to Docker-Deployment directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

print_status "Current directory: $(pwd)"
echo ""

# Clean up old containers and images
print_status "Step 1: Cleaning up old containers and images..."
docker-compose down 2>/dev/null || true
docker system prune -f

echo ""
print_status "Step 2: Building ConCare-RL Docker image..."
if docker-compose build --no-cache; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Build completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
else
    echo ""
    print_error "Docker build failed!"
    print_error "Check the error messages above for details."
    exit 1
fi

# Ask user what to do next
echo "What would you like to do next?"
echo "[1] Run development environment (port 5000)"
echo "[2] Run production environment (port 80)"  
echo "[3] Just build (exit)"
echo "[4] Run with monitoring (Prometheus + Grafana)"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo ""
        print_status "Starting development environment..."
        if docker-compose up -d; then
            echo ""
            echo -e "${GREEN}========================================${NC}"
            echo -e "${GREEN}Development environment started!${NC}"
            echo -e "${GREEN}========================================${NC}"
            echo -e "${BLUE}Web interface:${NC} http://localhost:5000"
            echo -e "${BLUE}Health check:${NC} http://localhost:5000/health"
            echo ""
            echo -e "${YELLOW}Useful commands:${NC}"
            echo "  To view logs: docker-compose logs -f"
            echo "  To stop: docker-compose down"
        else
            print_error "Failed to start development environment"
            exit 1
        fi
        ;;
    2)
        echo ""
        print_status "Starting production environment..."
        if docker-compose -f docker-compose.prod.yml up -d; then
            echo ""
            echo -e "${GREEN}========================================${NC}"
            echo -e "${GREEN}Production environment started!${NC}"
            echo -e "${GREEN}========================================${NC}"
            echo -e "${BLUE}Web interface:${NC} http://localhost"
            echo -e "${BLUE}Health check:${NC} http://localhost/health"
            echo ""
            echo -e "${YELLOW}Useful commands:${NC}"
            echo "  To view logs: docker-compose -f docker-compose.prod.yml logs -f"
            echo "  To stop: docker-compose -f docker-compose.prod.yml down"
        else
            print_error "Failed to start production environment"
            exit 1
        fi
        ;;
    4)
        echo ""
        print_status "Starting with monitoring..."
        if docker-compose -f docker-compose.prod.yml --profile monitoring up -d; then
            echo ""
            echo -e "${GREEN}========================================${NC}"
            echo -e "${GREEN}Full stack started with monitoring!${NC}"
            echo -e "${GREEN}========================================${NC}"
            echo -e "${BLUE}Web interface:${NC} http://localhost"
            echo -e "${BLUE}Prometheus:${NC} http://localhost:9090"
            echo -e "${BLUE}Grafana:${NC} http://localhost:3000 (admin/admin)"
            echo ""
            echo -e "${YELLOW}Useful commands:${NC}"
            echo "  To view logs: docker-compose -f docker-compose.prod.yml logs -f"
            echo "  To stop: docker-compose -f docker-compose.prod.yml --profile monitoring down"
        else
            print_error "Failed to start monitoring stack"
            exit 1
        fi
        ;;
    *)
        echo ""
        print_status "Build completed. You can run the application later with:"
        echo "  docker-compose up -d (development)"
        echo "  docker-compose -f docker-compose.prod.yml up -d (production)"
        ;;
esac

echo "" 