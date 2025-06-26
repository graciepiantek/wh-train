
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.height = config['height']
        self.width = config['width']
        self.num_channels = config['num_channels']
        self.batch_size = config['batch_size']
        self.validation_split = config['validation_split']
        
    def create_data_generators(self):
        """Create data generators with augmentation settings from config"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=self.validation_split,
            width_shift_range=self.config.get('width_shift_range', 0.0),
            height_shift_range=self.config.get('height_shift_range', 0.0),
            horizontal_flip=self.config.get('horizontal_flip', False)
        )
        
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=self.validation_split
        )
        
        return train_datagen, val_datagen
    
    def load_data(self, data_path):
        """Load training and validation data from directory structure"""
        train_datagen, val_datagen = self.create_data_generators()
        
        train_generator = train_datagen.flow_from_directory(
            data_path,
            target_size=(self.height, self.width),
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training'
        )
        
        validation_generator = val_datagen.flow_from_directory(
            data_path,
            target_size=(self.height, self.width),
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation'
        )
        
        return train_generator, validation_generator
    
    def get_class_names(self, data_path):
        """Get class names from directory structure"""
        return sorted([d for d in os.listdir(data_path) 
                      if os.path.isdir(os.path.join(data_path, d))])
    
    def preprocess_image(self, image_path):
        """Preprocess single image for prediction"""
        image = tf.keras.preprocessing.image.load_img(
            image_path, 
            target_size=(self.height, self.width)
        )
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array /= 255.0
        return image_array
    