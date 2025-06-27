#!/bin/bash

if [ ! -d "venv" ]; then
    /usr/bin/python3 -m venv venv
fi

source venv/bin/activate

pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "Installing ML dependencies..."
    pip install tensorflow scikit-learn matplotlib seaborn h5py
fi

echo "Setup complete! Virtual environment is active."
echo "When finished, run: deactivate"