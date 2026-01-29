#!/bin/bash

# TinyBERT Service - Production Deployment Script with Gunicorn

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate deploy-ml

# Navigate to project directory
cd "$(dirname "$0")"

# Display configuration
echo "============================================"
echo "TinyBERT Service - Production Mode"
echo "============================================"
echo "Server: Gunicorn with Uvicorn workers"
echo "Port: 8000"
echo "Workers: Auto-calculated (CPU cores * 2 + 1)"
echo "Config: gunicorn_conf.py"
echo ""
echo "API will be available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo "============================================"
echo ""

# Start gunicorn with uvicorn workers
gunicorn -c gunicorn_conf.py app.main:app
