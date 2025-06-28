#!/bin/bash

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

pip install --upgrade pip
pip install tensorflow[and-cuda]==2.15.0

echo "Installing other dependencies..."
pip install scikit-learn==1.4.0 matplotlib==3.8.0 seaborn==0.12.0 h5py==3.10.0
pip install Pillow pandas numpy

echo "Setup complete! Virtual environment is active."
echo "To test GPU: python -c \"import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))\""
echo "When finished, run: deactivate"