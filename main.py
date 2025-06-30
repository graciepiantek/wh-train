import os
# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

import argparse
import json
import tensorflow as tf
import time
from data_loader import DataLoader
from evaluator import Evaluator
from model_loader import ModelLoader

def main():
    parser = argparse.ArgumentParser(description='Transfer Learning Evaluation')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--model_type', type=str, choices=['original', 'efficientnet', 'resnet'], 
                       required=True, help='Model type to evaluate')
    parser.add_argument('--weights_path', type=str, help='Path to pre-trained weights (for original model)')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    
    args = parser.parse_args()
    
    start_total = time.time()
    
    with open(args.config, 'r') as f:
        config = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    
    tf.config.set_visible_devices([], 'GPU')
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {len(gpus)} GPU(s)")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU available, using CPU")
    
    print(f"\n{'='*60}")
    print(f"CONFIGURATION")
    print(f"{'='*60}")
    print(f"Config file: {args.config}")
    print(f"Data path: {args.data_path}")
    print(f"Model type: {args.model_type}")
    print(f"Output directory: {args.output_dir}")
    print(f"Image dimensions: {config['height']}x{config['width']}x{config['num_channels']}")
    if args.weights_path:
        print(f"Weights path: {args.weights_path}")
    print(f"{'='*60}\n")
    
    print("Loading data...")
    data_start = time.time()
    data_loader = DataLoader(config)
    train_data, val_data = data_loader.load_data(args.data_path)
    data_time = time.time() - data_start
    print(f"Data loading completed in {data_time/60:.1f} minutes")
    
    model_loader = ModelLoader(config)
    
    if args.model_type == 'original':
        if not args.weights_path:
            raise ValueError("--weights_path required for original models")
        
        print(f"\n{'='*40}")
        print(f"LOADING ORIGINAL MODEL")
        print(f"{'='*40}")
        print(f"Loading original model from: {args.weights_path}")
        
        model_start = time.time()
        model = model_loader.load_model_by_type(
            args.model_type, 
            weights_path=args.weights_path,
            intersection=config.get('intersection')
        )
        model_time = time.time() - model_start
        
        if model is None:
            raise ValueError("Failed to load original model")
        
        print(f"Original model loaded successfully in {model_time:.1f} seconds")
        print(f"Model has {model.count_params():,} total parameters")
        
    elif args.model_type in ['efficientnet', 'resnet']:
        print(f"\n{'='*40}")
        print(f"CREATING {args.model_type.upper()} MODEL")
        print(f"{'='*40}")
        
        model_start = time.time()
        model = model_loader.load_model_by_type(args.model_type)
        model_time = time.time() - model_start
        
        print(f"{args.model_type} model created in {model_time:.1f} seconds")
        
        epochs = config['epochs']
        patience = config.get('patience', 5)
        
        print(f"\n{'='*40}")
        print(f"TRAINING {args.model_type.upper()}")
        print(f"{'='*40}")
        print(f"Training for {epochs} epochs with patience {patience}")
        print(f"Starting training at {time.strftime('%H:%M:%S')}")
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=patience,
                restore_best_weights=True,
                monitor='val_accuracy',
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=patience//2,
                min_lr=1e-7,
                monitor='val_loss',
                verbose=1
            )
        ]
        
        train_start = time.time()
        history = model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1
        )
        train_time = time.time() - train_start
        
        print(f"\nTraining completed at {time.strftime('%H:%M:%S')}")
        print(f"Training took {train_time/60:.1f} minutes")
        print(f"Completed {len(history.history['loss'])} epochs")
        
        training_history = {
            'model_type': args.model_type,
            'config_used': config,
            'epochs_completed': len(history.history['loss']),
            'epochs_planned': epochs,
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'training_time_minutes': train_time/60,
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
    
    print(f"\n{'='*40}")
    print(f"EVALUATING MODEL")
    print(f"{'='*40}")
    
    eval_start = time.time()
    evaluator = Evaluator(config, args.output_dir)
    results = evaluator.evaluate_model(model, val_data, args.model_type)
    evaluator.save_results(results, args.model_type)
    eval_time = time.time() - eval_start

    total_time = time.time() - start_total
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {args.model_type}")
    print(f"Final Validation Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Final Validation Loss: {results['loss']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print(f"Total Parameters: {results['total_params']:,}")
    print(f"Inference Time per Image: {results['inference_time_per_image']:.6f}s")
    
    print(f"\n{'='*60}")
    print(f"TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"Data loading: {data_time/60:.1f} minutes ({data_time/total_time*100:.1f}%)")
    if args.model_type in ['efficientnet', 'resnet']:
        print(f"Training: {train_time/60:.1f} minutes ({train_time/total_time*100:.1f}%)")
    print(f"Evaluation: {eval_time/60:.1f} minutes ({eval_time/total_time*100:.1f}%)")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Completed at: {time.strftime('%H:%M:%S')}")
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Evaluation complete!")

if __name__ == "__main__":
    main()