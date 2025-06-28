import argparse
import json
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

def load_and_plot_training_history(history_file):
    with open(history_file, 'r') as f:
        data = json.load(f)
    
    history = data['history']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(history['accuracy'], label='Training Accuracy', color='blue')
    axes[0, 0].plot(history['val_accuracy'], label='Validation Accuracy', color='red')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['loss'], label='Training Loss', color='blue')
    axes[0, 1].plot(history['val_loss'], label='Validation Loss', color='red')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    metrics = {
        'Final Train Acc': data['final_train_accuracy'],
        'Final Val Acc': data['final_val_accuracy'],
        'Final Train Loss': data['final_train_loss'],
        'Final Val Loss': data['final_val_loss']
    }
    
    axes[1, 0].bar(range(len(metrics)), list(metrics.values()), 
                   color=['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'])
    axes[1, 0].set_xticks(range(len(metrics)))
    axes[1, 0].set_xticklabels(list(metrics.keys()), rotation=45)
    axes[1, 0].set_title('Final Metrics')
    axes[1, 0].set_ylabel('Value')
    
    info_text = f"""
    Model: {data['model_type']}
    Epochs Completed: {data['epochs_completed']}/{data['epochs_planned']}
    Best Val Accuracy: {max(history['val_accuracy']):.4f}
    Best Val Loss: {min(history['val_loss']):.4f}
    Config: {data['config_used']['intersection']} intersection
    Learning Rate: {data['config_used']['learning_rate']}
    Batch Size: {data['config_used']['batch_size']}
    """
    
    axes[1, 1].text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center')
    axes[1, 1].set_title('Training Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()

    os.makedirs('analysis_results', exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'analysis_results/efficientnet_analysis_{timestamp}'
    
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{filename}.pdf', dpi=300, bbox_inches='tight')

    plt.show()
    
    print(f"Final Validation Accuracy: {data['final_val_accuracy']:.4f} ({data['final_val_accuracy']*100:.2f}%)")
    print(f"Final Validation Loss: {data['final_val_loss']:.4f}")
    print(f"Training completed in {data['epochs_completed']} epochs")
    print(f"Peak validation accuracy: {max(history['val_accuracy']):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze training results')
    parser.add_argument('--history_file', type=str, required=True, 
                       help='Path to training history JSON file')
    args = parser.parse_args()
    
    load_and_plot_training_history(args.history_file)