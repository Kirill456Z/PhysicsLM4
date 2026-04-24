#!/bin/bash
set -e

echo "========================================"
echo "Installing Jupyter Notebook..."
echo "========================================"
pip install jupyter

echo "========================================"
echo "Starting Jupyter Notebook server..."
echo "========================================"
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
