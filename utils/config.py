
import os

class Config:
    input_shape = (180, 170, 3) 

    data_path = os.path.expanduser('~/Downloads/train_dataset.h5')
    pretrained_models_path = os.path.expanduser('~/Downloads/pretrained_models/')

    batch_size = 32
    epochs = 50
    learning_rate = 1e-4
    validation_split = 0.2
    
    few_shot_epochs = 20
    few_shot_LR = 1e-5
    few_shot_samples = [5, 10, 20]

    models_to_compare = [
        'original_fourth',
        'original_chestnut',
        'efficientnet_frozen',
        'efficientnet_finetuned',
        'resnet_frozen',
        'yolo_pretrained'
    ]

    include_pretrained_models = True
    
    results_path = 'results/'
    models_save_path = 'saved_models/'

