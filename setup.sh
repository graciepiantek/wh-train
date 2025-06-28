#!/bin/bash

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

echo "Activating virtual environment"
source venv/bin/activate

pip install --upgrade pip

echo "Current GPU status:"
nvidia-smi

echo "Installing TensorFlow with CUDA 12.x support"
pip install tensorflow==2.15.0

echo "Installing other dependencies..."
pip install scikit-learn==1.4.0 matplotlib==3.8.0 seaborn==0.12.0 h5py==3.10.0
pip install Pillow pandas numpy

echo "Testing GPU availability..."
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU Available:', tf.config.list_physical_devices('GPU')); print('Built with CUDA:', tf.test.is_built_with_cuda())"

echo "Setup complete! Virtual environment is active."
echo "When finished, run: deactivate"