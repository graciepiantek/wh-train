import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetV2S, ResNet152V2
from tensorflow.keras.models import Model
import h5py
import json
from pathlib import Path

#new method version and keras file type for all models
class ModelLoader:
    def __init__(self, config=None):
        self.config = config
        if config:
            self.height = config['height']
            self.width = config['width']
            self.num_channels = config['num_channels']
            self.learning_rate = config.get('learning_rate', 1e-3)
    
    @staticmethod
    def load_legacy_model(model_path):
        with h5py.File(model_path, 'r') as f:
            model_config_str = f.attrs['model_config']
            if isinstance(model_config_str, bytes):
                model_config_str = model_config_str.decode('utf-8')
            model_config = json.loads(model_config_str)
        
        model = Sequential()
        for i, layer_config in enumerate(model_config['config']['layers']):
            layer_class = layer_config['class_name']
            layer_params = layer_config['config']
            
            if layer_class == 'Conv2D':
                if i == 0:
                    input_shape = layer_params.get('batch_input_shape', [None, 216, 384, 3])[1:]
                    model.add(Conv2D(layer_params['filters'], layer_params['kernel_size'],
                                   activation=layer_params['activation'], padding=layer_params['padding'],
                                   input_shape=input_shape))
                else:
                    model.add(Conv2D(layer_params['filters'], layer_params['kernel_size'],
                                   activation=layer_params['activation'], padding=layer_params['padding']))
            elif layer_class == 'MaxPooling2D':
                model.add(MaxPooling2D(layer_params['pool_size'], strides=layer_params['strides']))
            elif layer_class == 'Dropout':
                model.add(Dropout(layer_params['rate']))
            elif layer_class == 'Flatten':
                model.add(Flatten())
            elif layer_class == 'Dense':
                model.add(Dense(layer_params['units'], activation=layer_params['activation']))
        
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
        return model
    
    @staticmethod
    def load_all_legacy_models(models_folder):
        models = {}
        models_path = Path(models_folder)
        
        for folder_name in ['modern_models', 'chestnut_original', 'fourth_original']:
            folder_path = models_path / folder_name
            if folder_path.exists():
                for model_file in list(folder_path.glob("**/*.keras")) + list(folder_path.glob("**/*.hdf5")):
                    model_name = str(model_file.relative_to(models_path).with_suffix('')).replace('/', '_')
                    try:
                        if model_file.suffix == '.keras':
                            models[model_name] = tf.keras.models.load_model(str(model_file))
                        else:
                            models[model_name] = ModelLoader.load_legacy_model(str(model_file))
                        print(f"Loaded legacy model: {model_name}")
                    except Exception as e:
                        print(f"Failed to load {model_name}: {e}")
                        continue
        return models
    
    def load_efficientnet(self, num_classes=None, trainable=False):
        if not self.config:
            raise ValueError("Config required for loading new models")
        
        base_model = EfficientNetV2S(
            weights='imagenet',
            include_top=False,
            input_shape=(self.height, self.width, self.num_channels)
        )
        
        base_model.trainable = trainable
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.config.get('dropout_rate', 0.2))(x)
        predictions = Dense(1, activation='sigmoid', name='predictions')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        model.compile(
            optimizer=self._get_optimizer(),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def load_resnet(self, num_classes=None, trainable=False):
        """Load ResNet152V2 with transfer learning setup"""
        if not self.config:
            raise ValueError("Config required for loading new models")
        
        base_model = ResNet152V2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.height, self.width, self.num_channels)
        )
        
        base_model.trainable = trainable
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.config.get('dropout_rate', 0.2))(x)
        predictions = Dense(1, activation='sigmoid', name='predictions')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        model.compile(
            optimizer=self._get_optimizer(),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _get_optimizer(self):
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        
        if optimizer_name == 'sgd':
            return tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate,
                momentum=self.config.get('momentum', 0.9)
            )
        elif optimizer_name == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        else:
            return tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    
    def load_model_by_type(self, model_type, num_classes=None, weights_path=None):
        if model_type == 'legacy' or model_type == 'original':
            if not weights_path:
                raise ValueError("weights_path required for legacy models")
            
            model = tf.keras.models.load_model(weights_path, compile=False)
            
            model.compile(
                optimizer='sgd',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
        
        elif model_type == 'efficientnet':
            return self.load_efficientnet(num_classes)
        
        elif model_type == 'resnet':
            return self.load_resnet(num_classes)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")