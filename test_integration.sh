#!/bin/bash

echo "=========================================="
echo "Testing Backend-Frontend Integration"
echo "=========================================="
echo ""

# Test 1: Check if backend is accessible
echo "Test 1: Checking backend health..."
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/health 2>/dev/null)

if [ "$response" = "200" ]; then
    echo "✅ Backend is running and healthy"
else
    echo "❌ Backend is not accessible (HTTP $response)"
    echo "   Please start the backend: python backend/app.py"
    exit 1
fi

echo ""

# Test 2: Check API endpoints
echo "Test 2: Testing API endpoints..."

# Test risk endpoint
echo "  - Testing /api/v1/risk/current..."
risk_response=$(curl -s http://localhost:5000/api/v1/risk/current?location_id=default 2>/dev/null)
if echo "$risk_response" | grep -q "risk_score"; then
    echo "    ✅ Risk endpoint working"
else
    echo "    ❌ Risk endpoint failed"
fi

# Test heatmap endpoint
echo "  - Testing /api/v1/map/heatmap..."
heatmap_response=$(curl -s http://localhost:5000/api/v1/map/heatmap 2>/dev/null)
if echo "$heatmap_response" | grep -q "grid_points"; then
    echo "    ✅ Heatmap endpoint working"
else
    echo "    ❌ Heatmap endpoint failed"
fi

# Test trends endpoint
echo "  - Testing /api/v1/risk/trends..."
trends_response=$(curl -s http://localhost:5000/api/v1/risk/trends?location_id=default 2>/dev/null)
if echo "$trends_response" | grep -q "current_window"; then
    echo "    ✅ Trends endpoint working"
else
    echo "    ❌ Trends endpoint failed"
fi

echo ""
echo "=========================================="
echo "Integration Test Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Start backend: python backend/app.py"
echo "2. Start frontend: cd frontend && npm run dev"
echo "3. Open browser: http://localhost:5173"
echo ""
