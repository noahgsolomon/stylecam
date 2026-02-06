#!/bin/bash

echo "stylecam setup"
echo "=============="
echo

# Check for OBS
if ! command -v /Applications/OBS.app/Contents/MacOS/OBS &> /dev/null; then
    echo "Installing OBS..."
    brew install --cask obs
    echo
    echo "OBS installed. Please:"
    echo "1. Open OBS"
    echo "2. Go to Tools -> Start Virtual Camera"
    echo "3. Close OBS"
    echo "4. Run this script again"
    echo
    exit 0
fi

echo "OBS found!"
echo

# Create virtual environment
echo "Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

echo
echo "Setup complete!"
echo
echo "To run:"
echo "  source venv/bin/activate"
echo "  FAL_KEY=your-key-here python stylecam.py"
echo
