#!/bin/bash

# Document Chunk Service Startup Script
# Usage: ./start_server.sh [config_file] [host] [port]

# Default values
CONFIG_FILE="${1:-../config/bge_m3_400.yaml}"
HOST="${2:-0.0.0.0}"
PORT="${3:-8000}"

echo "🚀 Starting Document Chunk Service"
echo "📝 Config: $CONFIG_FILE"
echo "🌐 Host: $HOST"
echo "🔌 Port: $PORT"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    echo "Available configs in ../config/:"
    ls -1 ../config/*.yaml 2>/dev/null || echo "No config files found"
    exit 1
fi

# Set environment variable and start server
export CHUNK_SERVICE_CONFIG="$CONFIG_FILE"
echo "🔧 Set CHUNK_SERVICE_CONFIG=$CHUNK_SERVICE_CONFIG"
echo ""

# Start the server with uvicorn CLI
echo "Starting server with uvicorn..."
uvicorn main:app --host "$HOST" --port "$PORT" --reload

# Alternative: you can also use python directly
# python main.py --config "$CONFIG_FILE" --host "$HOST" --port "$PORT" --reload