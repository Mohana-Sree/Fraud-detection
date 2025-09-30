#!/bin/bash
# Build script for deployment platforms
echo "🔨 Starting deployment build..."

# Install Python dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Setup model (train if needed)
echo "🤖 Setting up ML model..."
python setup_deployment.py

echo "✅ Build completed successfully!"