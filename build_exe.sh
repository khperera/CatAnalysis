#!/usr/bin/env bash
# One-command build script for CatDetector exe
# Usage: bash build_exe.sh

set -e

echo "=== Installing dependencies ==="
pip install -r requirements.txt
pip install pyinstaller

echo "=== Building exe ==="
pyinstaller CatApp.spec --clean

echo ""
echo "=== Done ==="
echo "Executable is in:  dist/CatDetector/"
echo "Run it with:       dist/CatDetector/CatDetector"
