import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetV2S, ResNet152V2
import h5py
import json
from pathlib import Path

class TrainDetectionModel:
    @staticmethod
    def build(width, height, num_channels):
        conv_kernel_size = (5, 5)
        model = Sequential()
        model.add(
            Conv2D(8,
                   conv_kernel_size,
                   input_shape=(height, width, num_channels),
                   activation='relu',
                   padding='same'))
        model.add(MaxPooling2D())
        model.add(Dropout(0.2))
        model.add(
            Conv2D(16, conv_kernel_size, activation='relu', padding='same'))
        model.add(MaxPooling2D())
        model.add(Dropout(0.2))
        model.add(
            Conv2D(32, conv_kernel_size, activation='relu', padding='same'))
        model.add(MaxPooling2D())
        model.add(Dropout(0.2))
        model.add(
            Conv2D(64, conv_kernel_size, activation='relu', padding='same'))
        model.add(MaxPooling2D())
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        return model

class SignalDetectionModel:
    @staticmethod
    def build(intersection, width, height, num_channels):
        model = Sequential()
        if intersection == 'fourth':
            model.add(
                Conv2D(8, (3, 3),
                       input_shape=(height, width, num_channels),
                       activation='relu',
                       padding='same'))
            model.add(MaxPooling2D())
            model.add(Dropout(0.2))
            model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D())
            model.add(Dropout(0.2))
            model.add(Flatten())
            model.add(Dense(16, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))
        elif intersection == 'chestnut':
            max_pool_size = (2, 2)
            max_pool_stride = 2
            conv_kernel_size = (3, 3)
            model.add(
                Conv2D(32, (5, 5),
                       input_shape=(height, width, num_channels),
                       activation='relu'))
            model.add(MaxPooling2D(pool_size=(3, 3), strides=max_pool_stride))
            model.add(Dropout(0.2))
            model.add(Conv2D(32, conv_kernel_size, activation='relu'))
            model.add(
                MaxPooling2D(pool_size=max_pool_size, strides=max_pool_stride))
            model.add(Dropout(0.2))
            model.add(Conv2D(64, conv_kernel_size, activation='relu'))
            model.add(
                MaxPooling2D(pool_size=max_pool_size, strides=max_pool_stride))
            model.add(Dropout(0.2))
            model.add(Flatten())
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))
        return model

class ModelLoader:
    def __init__(self, config=None):
        self.config = config
        if config:
            self.height = config['height']
            self.width = config['width']
            self.num_channels = config['num_channels']
    
    def load_original_model(self, model_type, intersection=None):
        if model_type == 'train' or 'train' in str(model_type):
            model = TrainDetectionModel.build(
                width=self.width, 
                height=self.height, 
                num_channels=self.num_channels
            )
        elif model_type == 'signal' or 'signal' in str(model_type):
            if intersection is None:
                intersection = self.config.get('intersection')
            if intersection is None:
                raise ValueError("intersection parameter required for signal models")
            model = SignalDetectionModel.build(
                intersection=intersection,
                width=self.width,
                height=self.height, 
                num_channels=self.num_channels
            )
        else:
            raise ValueError(f"Unknown original model type: {model_type}")
        
        optimizer = self._get_optimizer(use_original_config=True)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
def load_legacy_model_with_weights(self, model_path, config=None):
    try:
        if model_path.endswith('.keras'):
            model = tf.keras.models.load_model(model_path, compile=False)

            optimizer = self._get_optimizer(use_original_config=True)
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            return model

        elif model_path.endswith(('.hdf5', '.h5')):
            intersection = self.config.get('intersection')
            
            if intersection in ['chestnut', 'fourth']:
                model = self.load_original_model('signal', intersection)
            else:
                model = self.load_original_model('train')
            
            model.load_weights(model_path)
            return model
            
    except Exception as e:
        print(f"Error loading legacy model {model_path}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    def load_efficientnet(self):
        if not self.config:
            raise ValueError("Config required for loading new models")
        
        base_model = EfficientNetV2S(
            weights='imagenet',
            include_top=False,
            input_shape=(self.height, self.width, self.num_channels)
        )

        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.config.get('dropout_rate', 0.2))(x)
        predictions = Dense(1, activation='sigmoid', name='predictions')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)

        optimizer = self._get_optimizer(use_original_config=False)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"EfficientNet created with {model.count_params():,} total parameters")
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        print(f"Only {trainable_params:,} parameters are trainable (classification head only)")
        
        return model

    def load_resnet(self):
        if not self.config:
            raise ValueError("Config required for loading new models")
        
        base_model = ResNet152V2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.height, self.width, self.num_channels)
        )

        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.config.get('dropout_rate', 0.2))(x)
        predictions = Dense(1, activation='sigmoid', name='predictions')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        optimizer = self._get_optimizer(use_original_config=False)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"ResNet created with {model.count_params():,} total parameters")
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        print(f"Only {trainable_params:,} parameters are trainable (classification head only)")
        
        return model
    
    def _get_optimizer(self, use_original_config=True):
        if use_original_config:
            optimizer_name = self.config.get('optimizer', 'SGD').upper()
            lr = self.config.get('learning_rate', 1e-3)
            
            if optimizer_name == 'SGD':
                return tf.keras.optimizers.SGD(
                    learning_rate=lr,
                    momentum=self.config.get('momentum', 0.9)
                )
            elif optimizer_name == 'ADAM':
                return tf.keras.optimizers.Adam(learning_rate=lr)
            else:
                return tf.keras.optimizers.SGD(
                    learning_rate=lr,
                    momentum=self.config.get('momentum', 0.9)
                )
        else:
            lr = self.config.get('transfer_learning_rate', 1e-4)
            return tf.keras.optimizers.Adam(learning_rate=lr)
    
    def load_model_by_type(self, model_type, weights_path=None, intersection=None):
        if model_type == 'original':
            if weights_path:
                return self.load_legacy_model_with_weights(weights_path, self.config)
            else:
                filename = "train" if intersection is None else "signal"
                return self.load_original_model(filename, intersection)
        elif model_type == 'efficientnet':
            return self.load_efficientnet()
        elif model_type == 'resnet':
            return self.load_resnet()
        else:
            raise ValueError(f"Unknown model type: {model_type}")