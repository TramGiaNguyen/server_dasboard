#!/bin/bash
# =============================================================================
# Cleanup Script for Smart Parking System
# =============================================================================
# Removes old database records and orphaned image files.
#
# Usage:
#   ./cleanup.sh --days 7              # Preview what will be deleted
#   ./cleanup.sh --days 7 --commit     # Actually delete
#   docker-compose -f docker-compose.gpu.yml exec parking python main.py cleanup --days 7 --dry-run
#   docker-compose -f docker-compose.gpu.yml exec parking python main.py cleanup --days 7
# =============================================================================

set -e

DAYS=7
COMMIT=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --days)
            DAYS="$2"
            shift 2
            ;;
        --commit)
            COMMIT=true
            shift
            ;;
        --dry-run)
            COMMIT=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--days N] [--commit]"
            echo ""
            echo "Options:"
            echo "  --days N     Number of days to retain (default: 7)"
            echo "  --commit     Actually delete records and files (default is dry-run)"
            echo "  --dry-run    Show what would be deleted (default)"
            echo ""
            echo "Examples:"
            echo "  $0 --days 7             # Preview cleanup, keep 7 days"
            echo "  $0 --days 7 --commit    # Run cleanup, keep 7 days"
            echo "  $0 --days 30 --commit   # Run cleanup, keep 30 days"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "  Smart Parking Cleanup Script"
echo "=========================================="
echo "  Retention days: $DAYS"
if [ "$COMMIT" = true ]; then
    echo "  Mode: COMMIT (will delete data)"
else
    echo "  Mode: DRY-RUN (preview only)"
fi
echo "=========================================="

# Check if running inside Docker
if [ -f /.dockerenv ] || grep -q docker /proc/1/cgroup 2>/dev/null; then
    echo "Running inside Docker container."
    PYTHON_CMD="python"
else
    echo "Running on host system."
    # Try to find the virtual environment
    if [ -d "venv" ]; then
        PYTHON_CMD="venv/Scripts/python"
    elif [ -d ".venv" ]; then
        PYTHON_CMD=".venv/Scripts/python"
    else
        PYTHON_CMD="python"
    fi
fi

# Run cleanup
if [ "$COMMIT" = true ]; then
    $PYTHON_CMD main.py cleanup --days $DAYS
else
    $PYTHON_CMD main.py cleanup --days $DAYS --dry-run
fi
