#!/bin/bash
# Build script for Smart Parking System Docker images

set -e

echo "=========================================="
echo "Smart Parking System - Docker Build"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found!${NC}"
    echo "Creating .env from .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${GREEN}✓ Created .env file${NC}"
    else
        echo -e "${RED}Error: .env.example not found!${NC}"
        exit 1
    fi
fi

# Check if backend_app/.env exists
if [ ! -f backend_app/.env ]; then
    echo -e "${YELLOW}Warning: backend_app/.env file not found!${NC}"
    echo "Creating backend_app/.env..."
    cat > backend_app/.env << EOF
APP_PORT=5002
FLASK_ENV=production
SECRET_KEY=smart-parking-backend-secret-key-2024-change-in-production

DB_HOST=postgres
DB_PORT=5432
DB_NAME=PARKING_PLATE
DB_USER=postgres
DB_PASSWORD=1412
EOF
    echo -e "${GREEN}✓ Created backend_app/.env file${NC}"
fi

# Parse command line arguments
BUILD_TARGET="runtime-cpu"
NO_CACHE=""
PULL=""
GPU_MODE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            BUILD_TARGET="runtime-gpu"
            GPU_MODE="-f docker-compose.gpu.yml"
            shift
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --pull)
            PULL="--pull"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--gpu] [--no-cache] [--pull]"
            exit 1
            ;;
    esac
done

echo "Build configuration:"
echo "  Target: $BUILD_TARGET"
echo "  GPU Mode: ${GPU_MODE:-disabled}"
echo "  No cache: ${NO_CACHE:-false}"
echo "  Pull base images: ${PULL:-false}"
echo ""

# Stop existing containers
echo "1. Stopping existing containers..."
docker-compose down 2>/dev/null || true
echo -e "${GREEN}✓ Containers stopped${NC}"
echo ""

# Build images
echo "2. Building Docker images..."
echo "   This may take 5-10 minutes on first build..."
if [ -n "$GPU_MODE" ]; then
    docker-compose -f docker-compose.yml -f docker-compose.gpu.yml build $NO_CACHE $PULL
else
    docker-compose build $NO_CACHE $PULL
fi
echo -e "${GREEN}✓ Images built successfully${NC}"
echo ""

# Start services
echo "3. Starting services..."
if [ -n "$GPU_MODE" ]; then
    docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
else
    docker-compose up -d
fi
echo -e "${GREEN}✓ Services started${NC}"
echo ""

# Wait for services to be healthy
echo "4. Waiting for services to be healthy..."
sleep 5

# Check service status
echo ""
echo "Service Status:"
docker-compose ps
echo ""

# Show logs
echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo ""
echo "Services:"
echo "  Dashboard:     http://localhost:5001"
echo "  Backend API:   http://localhost:5002"
echo "  PostgreSQL:    localhost:5432"
echo ""
if [ -n "$GPU_MODE" ]; then
    echo "GPU Support: ENABLED (CUDA 12.8)"
    echo ""
fi
echo "View logs:"
echo "  docker-compose logs -f parking"
echo "  docker-compose logs -f backend"
echo "  docker-compose logs -f postgres"
echo ""
echo "Stop services:"
echo "  docker-compose down"
echo ""
