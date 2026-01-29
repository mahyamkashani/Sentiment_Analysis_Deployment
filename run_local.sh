#!/bin/bash

# TinyBERT Service - Local Deployment Script

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate deploy-ml

# Navigate to project directory
cd "$(dirname "$0")"

# Start uvicorn server
echo "Starting TinyBERT Service (Refactored)..."
echo "Server will be available at: http://localhost:8001"
echo "API Documentation: http://localhost:8001/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
