import tensorflow as tf
import numpy as np
import json
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class Evaluator:
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        
    def evaluate_model(self, model, validation_data, model_name):
        print(f"Evaluating {model_name} model...")
        
        predictions = model.predict(validation_data, verbose=1)
        
        if len(predictions.shape) == 2 and predictions.shape[1] == 1:
            predicted_classes = (predictions > 0.5).astype(int).flatten()
        else:
            predicted_classes = np.argmax(predictions, axis=1)
        
        true_classes = validation_data.classes
        class_labels = list(validation_data.class_indices.keys())
        
        loss, accuracy = model.evaluate(validation_data, verbose=0)
        
        report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=class_labels,
            output_dict=True,
            zero_division=0
        )
        
        cm = confusion_matrix(true_classes, predicted_classes)
        
        results = {
            'model_name': model_name,
            'loss': float(loss),
            'accuracy': float(accuracy),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'class_labels': class_labels
        }
        
        return results
    
    def save_results(self, results, model_name):
        json_path = os.path.join(self.output_dir, f'{model_name}_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.plot_confusion_matrix(
            results['confusion_matrix'], 
            results['class_labels'], 
            model_name
        )
        
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Loss: {results['loss']:.4f}")
        
    def plot_confusion_matrix(self, cm, class_labels, model_name):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_labels, yticklabels=class_labels)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plot_path = os.path.join(self.output_dir, f'{model_name}_confusion_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def compare_models(self, results_list):
        comparison = {}
        for results in results_list:
            comparison[results['model_name']] = {
                'accuracy': results['accuracy'],
                'loss': results['loss']
            }
        
        comparison_path = os.path.join(self.output_dir, 'model_comparison.json')
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        return comparison