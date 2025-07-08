#!/bin/bash

# ConCare-RL Docker Cleanup Script
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ConCare-RL Docker Cleanup Script${NC}"
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
    exit 1
fi

# Navigate to Docker-Deployment directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

echo "What would you like to clean up?"
echo "[1] Stop all ConCare-RL containers"
echo "[2] Remove ConCare-RL containers (keeps images)"
echo "[3] Remove ConCare-RL images (keeps volumes)"
echo "[4] Remove all ConCare-RL data (containers, images, volumes)"
echo "[5] Clean Docker system (removes all unused data)"
echo "[6] Full reset (stop everything and clean all Docker data)"
echo ""
read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        print_status "Stopping all ConCare-RL containers..."
        docker-compose down 2>/dev/null || true
        docker-compose -f docker-compose.prod.yml down 2>/dev/null || true
        docker-compose -f docker-compose.prod.yml --profile monitoring down 2>/dev/null || true
        print_status "All ConCare-RL containers stopped."
        ;;
    2)
        print_status "Removing ConCare-RL containers..."
        docker-compose down --remove-orphans 2>/dev/null || true
        docker-compose -f docker-compose.prod.yml down --remove-orphans 2>/dev/null || true
        docker-compose -f docker-compose.prod.yml --profile monitoring down --remove-orphans 2>/dev/null || true
        
        # Remove any remaining ConCare-RL containers
        CONTAINERS=$(docker ps -a --filter "name=conmedrl" --format "{{.ID}}")
        if [ ! -z "$CONTAINERS" ]; then
            echo "$CONTAINERS" | xargs docker rm -f
            print_status "Removed ConCare-RL containers."
        else
            print_status "No ConCare-RL containers found."
        fi
        ;;
    3)
        print_status "Removing ConCare-RL images..."
        # Stop containers first
        docker-compose down 2>/dev/null || true
        docker-compose -f docker-compose.prod.yml --profile monitoring down 2>/dev/null || true
        
        # Remove ConCare-RL images
        IMAGES=$(docker images --filter "reference=*conmedrl*" --format "{{.ID}}")
        if [ ! -z "$IMAGES" ]; then
            echo "$IMAGES" | xargs docker rmi -f
            print_status "Removed ConCare-RL images."
        else
            print_status "No ConCare-RL images found."
        fi
        
        # Remove related images
        docker rmi $(docker images --filter "dangling=true" -q) 2>/dev/null || true
        ;;
    4)
        print_warning "This will remove ALL ConCare-RL containers, images, and volumes!"
        read -p "Are you sure? (y/N): " confirm
        if [[ $confirm =~ ^[Yy]$ ]]; then
            print_status "Performing full ConCare-RL cleanup..."
            
            # Stop and remove everything
            docker-compose down --volumes --remove-orphans 2>/dev/null || true
            docker-compose -f docker-compose.prod.yml --profile monitoring down --volumes --remove-orphans 2>/dev/null || true
            
            # Remove ConCare-RL containers
            CONTAINERS=$(docker ps -a --filter "name=conmedrl" --format "{{.ID}}")
            if [ ! -z "$CONTAINERS" ]; then
                echo "$CONTAINERS" | xargs docker rm -f
            fi
            
            # Remove ConCare-RL images
            IMAGES=$(docker images --filter "reference=*conmedrl*" --format "{{.ID}}")
            if [ ! -z "$IMAGES" ]; then
                echo "$IMAGES" | xargs docker rmi -f
            fi
            
            # Remove ConCare-RL volumes
            VOLUMES=$(docker volume ls --filter "name=*conmedrl*" --format "{{.Name}}")
            if [ ! -z "$VOLUMES" ]; then
                echo "$VOLUMES" | xargs docker volume rm
            fi
            
            print_status "Full ConCare-RL cleanup completed."
        else
            print_status "Cleanup cancelled."
        fi
        ;;
    5)
        print_warning "This will clean up ALL unused Docker data (not just ConCare-RL)!"
        read -p "Are you sure? (y/N): " confirm
        if [[ $confirm =~ ^[Yy]$ ]]; then
            print_status "Cleaning Docker system..."
            docker system prune -a --volumes -f
            print_status "Docker system cleanup completed."
        else
            print_status "Cleanup cancelled."
        fi
        ;;
    6)
        print_error "WARNING: This will stop ALL Docker containers and remove ALL Docker data!"
        print_error "This affects ALL Docker applications on your system, not just ConCare-RL!"
        read -p "Are you absolutely sure? Type 'RESET' to confirm: " confirm
        if [[ $confirm == "RESET" ]]; then
            print_status "Performing full Docker reset..."
            
            # Stop all containers
            docker stop $(docker ps -q) 2>/dev/null || true
            
            # Remove all containers
            docker rm $(docker ps -aq) 2>/dev/null || true
            
            # Remove all images
            docker rmi $(docker images -q) 2>/dev/null || true
            
            # Remove all volumes
            docker volume rm $(docker volume ls -q) 2>/dev/null || true
            
            # Remove all networks
            docker network rm $(docker network ls -q) 2>/dev/null || true
            
            # Clean system
            docker system prune -a --volumes -f
            
            print_status "Full Docker reset completed."
        else
            print_status "Reset cancelled."
        fi
        ;;
    *)
        print_error "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
print_status "Cleanup completed!"
echo "" 