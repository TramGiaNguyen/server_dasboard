#!/bin/bash
# Script to run the mobile app backend server

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Run with gunicorn for production
if [ "$FLASK_ENV" = "production" ]; then
    echo "Starting production server with gunicorn..."
    gunicorn -w 4 -b 0.0.0.0:${APP_PORT:-5001} --worker-class eventlet -k eventlet app:app
else
    echo "Starting development server..."
    python app.py
fi
