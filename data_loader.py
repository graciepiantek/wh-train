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

    def load_data(self, data_path, use_temporal_split=True):
        if use_temporal_split:
            return self._load_data_with_temporal_split(data_path)
        else:
            return self._load_data_original(data_path)

    def _temporal_split_files_balanced(self, file_class_pairs, val_split=0.15, seed=42):
        
        print(f"Using simple random split for {len(file_class_pairs)} files")
        
        if len(file_class_pairs) < 2:
            print("Not enough files for split")
            return file_class_pairs, []
        
        train_files, val_files = train_test_split(
            file_class_pairs, 
            test_size=val_split, 
            random_state=seed,
            stratify=None
        )
        
        print(f"Split result: {len(train_files)} train, {len(val_files)} val")
        return train_files, val_files

    def _create_generator_from_files(self, file_class_pairs, augment=True):
        if not file_class_pairs:
            raise ValueError("No files provided for generator creation!")
        
        files, class_names = zip(*file_class_pairs)

        labels = []
        for class_name in class_names:
            if class_name in ['train', 'signal']:
                labels.append(1)
            else:
                labels.append(0)
        
        images = []
        valid_labels = []
        
        print(f"Loading {len(files)} images...")
        
        for i, (file_path, label) in enumerate(zip(files, labels)):
            try:
                img = load_img(file_path, target_size=(self.height, self.width))
                img_array = img_to_array(img)
                images.append(img_array)
                valid_labels.append(label)
                
                if (i + 1) % 1000 == 0:
                    print(f"Loaded {i + 1}/{len(files)} images...")
                    
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
                continue
        
        if not images:
            raise ValueError("No valid images found!")
        
        images = np.array(images)
        labels = np.array(valid_labels)
        
        print(f"Successfully loaded {len(images)} images")
        print(f"Class distribution: {np.sum(labels)} positive, {len(labels) - np.sum(labels)} negative")
        
        if augment:
            datagen = ImageDataGenerator(
                rescale=1./255,
                width_shift_range=self.config.get('width_shift_range', 0.0),
                height_shift_range=self.config.get('height_shift_range', 0.0),
                horizontal_flip=self.config.get('horizontal_flip', False),
                rotation_range=self.config.get('rotation_range', 0)
            )
        else:
            datagen = ImageDataGenerator(rescale=1./255)
        
        generator = datagen.flow(
            images, 
            labels,
            batch_size=self.batch_size,
            shuffle=True,
            seed=42
        )

        generator.samples = len(images)
        generator.classes = labels
        generator.class_indices = {'no_train': 0, 'train': 1} if any('train' in cn for cn in class_names) else {'no_signal': 0, 'signal': 1}
        
        return generator

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