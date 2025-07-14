#!/bin/bash

# ConMED-RL Research Environment Build Script
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ConMED-RL Research Environment Setup${NC}"
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
print_status "Cleaning up old containers and images..."
docker-compose -f docker-compose.research.yml down 2>/dev/null || true
docker system prune -f

echo ""
print_status "Building ConMED-RL Research Environment..."
docker-compose -f docker-compose.research.yml build --no-cache

if [ $? -ne 0 ]; then
    print_error "Docker build failed!"
    print_error "Check the error messages above for details."
    exit 1
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Build completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Ask user what to do next
echo "What would you like to do next?"
echo "[1] Start research environment (Jupyter Lab)"
echo "[2] Start development environment (Jupyter + Flask)"
echo "[3] Just build (exit)"
echo ""
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo ""
        print_status "Starting research environment..."
        docker-compose -f docker-compose.research.yml up -d
        if [ $? -eq 0 ]; then
            echo ""
            echo -e "${BLUE}========================================${NC}"
            echo -e "${BLUE}Research environment started!${NC}"
            echo -e "${BLUE}========================================${NC}"
            echo -e "${GREEN}Jupyter Lab: http://localhost:8888${NC}"
            echo -e "${GREEN}Token: conmed-rl-research${NC}"
            echo -e "${GREEN}Flask App: http://localhost:5000${NC}"
            echo ""
            echo -e "${YELLOW}Available directories:${NC}"
            echo "  - /app/ConMedRL/ - Core OCRL framework"
            echo "  - /app/Data/ - Data processing modules"
            echo "  - /app/Experiment Notebook/ - Jupyter notebooks"
            echo "  - /app/CDM-Software/ - Clinical decision support"
            echo ""
            echo "To view logs: docker-compose -f docker-compose.research.yml logs -f"
            echo "To stop: docker-compose -f docker-compose.research.yml down"
        fi
        ;;
    2)
        echo ""
        print_status "Starting development environment..."
        docker-compose -f docker-compose.dev.yml up -d
        if [ $? -eq 0 ]; then
            echo ""
            echo -e "${BLUE}========================================${NC}"
            echo -e "${BLUE}Development environment started!${NC}"
            echo -e "${BLUE}========================================${NC}"
            echo -e "${GREEN}Jupyter Lab: http://localhost:8888${NC}"
            echo -e "${GREEN}Token: conmed-rl-dev${NC}"
            echo -e "${GREEN}Flask App: http://localhost:5000${NC}"
            echo ""
            echo "To view logs: docker-compose -f docker-compose.dev.yml logs -f"
            echo "To stop: docker-compose -f docker-compose.dev.yml down"
        fi
        ;;
    *)
        echo ""
        print_status "Build completed. You can start the environment later with:"
        echo "  Research: docker-compose -f docker-compose.research.yml up -d"
        echo "  Development: docker-compose -f docker-compose.dev.yml up -d"
        ;;
esac

echo "" 