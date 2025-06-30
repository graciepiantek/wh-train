import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
import json
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
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {len(gpus)} GPU(s)")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU available, using CPU")
    
    data_loader = DataLoader(config)
    train_data, val_data = data_loader.load_data(args.data_path)
    
    model_loader = ModelLoader(config)
    model = model_loader.load_model_by_type(args.model_type, weights_path=args.weights_path)

    if args.model_type in ['efficientnet', 'resnet']:
        epochs = config['epochs']
        patience = config.get('patience', 5)
        
        print(f"Training {args.model_type} for {epochs} epochs with patience {patience}...")
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=patience,
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=patience//2,
                min_lr=1e-7,
                monitor='val_loss'
            )
        ]
        
        history = model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1
        )
        
        training_history = {
            'model_type': args.model_type,
            'config_used': config,
            'epochs_completed': len(history.history['loss']),
            'epochs_planned': epochs,
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'history': {
                'loss': [float(x) for x in history.history['loss']],
                'accuracy': [float(x) for x in history.history['accuracy']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']]
            }
        }
        
        history_file = os.path.join(args.output_dir, f'{args.model_type}_training_history.json')
        with open(history_file, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print(f"Training history saved to {history_file}")
        print(f"Training completed in {len(history.history['loss'])} epochs")
    
    evaluator = Evaluator(config, args.output_dir)
    results = evaluator.evaluate_model(model, val_data, args.model_type)
    evaluator.save_results(results, args.model_type)
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()