#!/bin/bash

source venv/bin/activate

mkdir -p results/{chestnut_signal,chestnut_train,fourth_signal,fourth_train}

CHESTNUT_SIGNAL_DATA="~/datasets/signal_chestnut"
CHESTNUT_TRAIN_DATA="~/datasets/train_chestnut" 
FOURTH_SIGNAL_DATA="~/datasets/signal_fourth"
FOURTH_TRAIN_DATA="~/datasets/train_fourth/hayden" 

CHESTNUT_SIGNAL_WEIGHTS="models/original/chestnut_signal/chestnut_signal_20200715_model.04-0.0102.keras"
CHESTNUT_TRAIN_WEIGHTS="models/original/chestnut_train/chestnut_train_20200731_model.20-0.0646.keras"
FOURTH_SIGNAL_WEIGHTS="models/original/fourth_signal/fourth_signal_20200803_model.08-0.1554.keras"
FOURTH_TRAIN_WEIGHTS="models/original/fourth_train/fourth_train_20200329_model.29-0.0538.keras"

for dir in "$CHESTNUT_SIGNAL_DATA" "$CHESTNUT_TRAIN_DATA" "$FOURTH_SIGNAL_DATA" "$FOURTH_TRAIN_DATA"; do
    if [ ! -d "$dir" ]; then
        echo "ERROR: Data directory $dir does not exist!"
        exit 1
    fi
done

for weights in "$CHESTNUT_SIGNAL_WEIGHTS" "$CHESTNUT_TRAIN_WEIGHTS" "$FOURTH_SIGNAL_WEIGHTS" "$FOURTH_TRAIN_WEIGHTS"; do
    if [ ! -f "$weights" ]; then
        echo "ERROR: Model weights $weights do not exist!"
        exit 1
    fi
done

echo "Starting Chestnut Signal evaluations..."
python main.py --config chestnut_signal_config.json --data_path $CHESTNUT_SIGNAL_DATA --model_type original --weights_path $CHESTNUT_SIGNAL_WEIGHTS --output_dir results/chestnut_signal
python main.py --config chestnut_signal_config.json --data_path $CHESTNUT_SIGNAL_DATA --model_type efficientnet --output_dir results/chestnut_signal
python main.py --config chestnut_signal_config.json --data_path $CHESTNUT_SIGNAL_DATA --model_type resnet --output_dir results/chestnut_signal

echo "Starting Fourth Signal evaluations..."
python main.py --config fourth_signal_config.json --data_path $FOURTH_SIGNAL_DATA --model_type original --weights_path $FOURTH_SIGNAL_WEIGHTS --output_dir results/fourth_signal
python main.py --config fourth_signal_config.json --data_path $FOURTH_SIGNAL_DATA --model_type efficientnet --output_dir results/fourth_signal
python main.py --config fourth_signal_config.json --data_path $FOURTH_SIGNAL_DATA --model_type resnet --output_dir results/fourth_signal

echo "Starting Chestnut Train evaluations..."
python main.py --config train_config.json --data_path $CHESTNUT_TRAIN_DATA --model_type original --weights_path $CHESTNUT_TRAIN_WEIGHTS --output_dir results/chestnut_train
python main.py --config train_config.json --data_path $CHESTNUT_TRAIN_DATA --model_type efficientnet --output_dir results/chestnut_train
python main.py --config train_config.json --data_path $CHESTNUT_TRAIN_DATA --model_type resnet --output_dir results/chestnut_train

echo "Starting Fourth Train evaluations..."
python main.py --config train_config.json --data_path $FOURTH_TRAIN_DATA --model_type original --weights_path $FOURTH_TRAIN_WEIGHTS --output_dir results/fourth_train
python main.py --config train_config.json --data_path $FOURTH_TRAIN_DATA --model_type efficientnet --output_dir results/fourth_train
python main.py --config train_config.json --data_path $FOURTH_TRAIN_DATA --model_type resnet --output_dir results/fourth_train

echo "All evaluations complete"