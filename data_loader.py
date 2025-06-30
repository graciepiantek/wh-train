import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import re

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.height = config['height']
        self.width = config['width']
        self.num_channels = config['num_channels']
        self.batch_size = config['batch_size']
        self.validation_split = config['validation_split']
        
    def create_data_generators(self):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=self.validation_split,
            width_shift_range=self.config.get('width_shift_range', 0.0),
            height_shift_range=self.config.get('height_shift_range', 0.0),
            horizontal_flip=self.config.get('horizontal_flip', False),
            rotation_range=self.config.get('rotation_range', 0)
        )
        
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=self.validation_split
        )
        
        return train_datagen, val_datagen

    def load_data(self, data_path):
        from sklearn.model_selection import train_test_split
        import os
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        import numpy as np
        
        contents = os.listdir(data_path)

        if 'signal' in contents and 'no_signal' in contents:
            class_dirs = {
                'signal': os.path.join(data_path, 'signal'),
                'no_signal': os.path.join(data_path, 'no_signal')
            }
        elif 'train' in contents and 'no_train' in contents:
            class_dirs = {
                'train': os.path.join(data_path, 'train'),
                'no_train': os.path.join(data_path, 'no_train')
            }
        else:
            raise ValueError(f"Unknown dataset structure: {contents}")
        
        all_files = []
        all_labels = []
        
        for class_name, class_path in class_dirs.items():
            files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for f in files:
                all_files.append(os.path.join(class_path, f))
                all_labels.append(1 if class_name in ['train', 'signal'] else 0)
        
        print(f"Found {len(all_files)} total files")
        
        train_files, val_files, train_labels, val_labels = train_test_split(
            all_files, all_labels, 
            test_size=0.15, 
            random_state=42,
            stratify=all_labels
        )
        
        print(f"Split: {len(train_files)} train, {len(val_files)} val")
        
        def load_images(files, labels):
            images = []
            valid_labels = []
            for file_path, label in zip(files, labels):
                try:
                    img = load_img(file_path, target_size=(self.height, self.width))
                    img_array = img_to_array(img) / 255.0
                    images.append(img_array)
                    valid_labels.append(label)
                except:
                    continue
            return np.array(images), np.array(valid_labels)
        
        print("Loading images")
        train_images, train_labels = load_images(train_files, train_labels)
        val_images, val_labels = load_images(val_files, val_labels)

        train_datagen = ImageDataGenerator(
            width_shift_range=self.config.get('width_shift_range', 0.0),
            height_shift_range=self.config.get('height_shift_range', 0.0),
            horizontal_flip=self.config.get('horizontal_flip', False),
            rotation_range=self.config.get('rotation_range', 0)
        )
        
        val_datagen = ImageDataGenerator()
        
        train_generator = train_datagen.flow(train_images, train_labels, batch_size=self.batch_size, shuffle=True)
        val_generator = val_datagen.flow(val_images, val_labels, batch_size=self.batch_size, shuffle=False)
        
        train_generator.samples = len(train_images)
        val_generator.samples = len(val_images)
        train_generator.classes = train_labels
        val_generator.classes = val_labels
        
        return train_generator, val_generator

    def _load_data_original(self, data_path):
        train_datagen, val_datagen = self.create_data_generators()
        
        train_generator = train_datagen.flow_from_directory(
            data_path,
            target_size=(self.height, self.width),
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training',
            shuffle=True,
            seed=42
        )
        
        validation_generator = val_datagen.flow_from_directory(
            data_path,
            target_size=(self.height, self.width),
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=True,
            seed=42
        )
        
        return train_generator, validation_generator
    
    def get_class_names(self, data_path):
        return sorted([d for d in os.listdir(data_path) 
                      if os.path.isdir(os.path.join(data_path, d))])
    
    def preprocess_image(self, image_path):
        image = load_img(image_path, target_size=(self.height, self.width))
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array /= 255.0
        return image_array