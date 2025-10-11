#!/bin/bash

# ConMED-RL Flask Application Launcher for Linux/Mac
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ConMED-RL Demo Application Launcher${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

echo -e "${YELLOW}NOTE: This is a demonstration version for testing and evaluation.${NC}"
echo -e "${YELLOW}For more information, see DEPLOYMENT_GUIDE.md${NC}"
echo ""

# Check if virtual environment should be activated
if [ -d "venv" ]; then
    echo -e "${GREEN}Found virtual environment, activating...${NC}"
    source venv/bin/activate
    echo ""
fi

# Function to check dependencies
check_dependencies() {
    local python_cmd=$1
    echo -e "${GREEN}Checking dependencies...${NC}"
    
    if $python_cmd -c "import flask, torch, numpy, sklearn, PIL" 2>/dev/null; then
        echo -e "${GREEN}Dependencies OK, starting application...${NC}"
        echo ""
        echo -e "${GREEN}Starting Flask app on http://localhost:5000${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop the application${NC}"
        echo ""
        $python_cmd web_application_demo.py
        return 0
    else
        echo -e "${RED}ERROR: Missing required dependencies!${NC}"
        echo -e "${YELLOW}Please install dependencies: $python_cmd -m pip install -r requirements.txt${NC}"
        echo ""
        return 1
    fi
}

echo -e "${GREEN}Checking Python installation...${NC}"
echo ""

# Try different Python commands
echo "Trying python..."
if command -v python &> /dev/null; then
    echo -e "${GREEN}Found python${NC}"
    if check_dependencies python; then
        exit 0
    else
        exit 1
    fi
fi

echo "Trying python3..."
if command -v python3 &> /dev/null; then
    echo -e "${GREEN}Found python3${NC}"
    if check_dependencies python3; then
        exit 0
    else
        exit 1
    fi
fi

echo -e "${RED}ERROR: Python not found in PATH!${NC}"
echo ""
echo "Please install Python or add it to your PATH variable."
echo "You can install Python from: https://www.python.org/downloads/"
echo ""
exit 1 