import argparse
import json
import os
import tensorflow as tf
from data_loader import DataLoader
from evaluator import Evaluator
from models.model_loader import ModelLoader

def main():
    parser = argparse.ArgumentParser(description='Transfer Learning Evaluation')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--model_type', type=str, choices=['original', 'efficientnet', 'resnet'], 
                       required=True, help='Model type to evaluate')
    parser.add_argument('--weights_path', type=str, help='Path to pre-trained weights (for original model)')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    
    data_loader = DataLoader(config)
    train_data, val_data = data_loader.load_data(args.data_path)
    
    model_loader = ModelLoader(config)
    model = model_loader.load_model_by_type(args.model_type, weights_path=args.weights_path)

    if args.model_type in ['efficientnet', 'resnet']:
        print(f"Training {args.model_type} for a few epochs...")
        
        history = model.fit(
            train_data,
            epochs=5, 
            validation_data=val_data,
            verbose=1
        )
    
    evaluator = Evaluator(config, args.output_dir)
    
    results = evaluator.evaluate_model(model, val_data, args.model_type)

    evaluator.save_results(results, args.model_type)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {len(gpus)} GPU(s)")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU available, using CPU")
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()