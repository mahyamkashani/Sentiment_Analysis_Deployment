#!/bin/bash

# Test script for TinyBERT Service API

BASE_URL="http://localhost:8001"

echo "========================================"
echo "Testing TinyBERT Service API"
echo "========================================"
echo ""

# Test 1: Root endpoint
echo "1. Testing root endpoint..."
curl -s $BASE_URL/ | python3 -m json.tool
echo ""

# Test 2: Health check
echo "2. Testing health check..."
curl -s $BASE_URL/api/v1/health | python3 -m json.tool
echo ""

# Test 3: Predict with full probabilities
echo "3. Testing predict endpoint..."
curl -s -X POST $BASE_URL/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I love this product!", "This is terrible"]}' \
  | python3 -m json.tool
echo ""

# Test 4: Predict top label
echo "4. Testing predict/top endpoint..."
curl -s -X POST $BASE_URL/api/v1/predict/top \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Amazing experience!", "Worst product ever"]}' \
  | python3 -m json.tool
echo ""

# Test 5: Batch predict
echo "5. Testing predict/batch endpoint..."
curl -s -X POST $BASE_URL/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["text1", "text2", "text3"]}' \
  | python3 -m json.tool
echo ""

echo "========================================"
echo "All tests completed!"
echo "========================================"
