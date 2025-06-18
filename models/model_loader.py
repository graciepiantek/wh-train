
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import h5py
import json
from pathlib import Path

class ModelLoader:
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
   def load_all_models(models_folder):
       models = {}
       models_path = Path(models_folder)
       
       for folder_name in ['weights', 'chestnut_original', 'fourth_original']:
           folder_path = models_path / folder_name
           if folder_path.exists():
               for model_file in folder_path.glob("**/*.hdf5"):
                   model_name = str(model_file.relative_to(models_path).with_suffix('')).replace('/', '_')
                   try:
                       models[model_name] = ModelLoader.load_legacy_model(str(model_file))
                   except:
                       continue
       
       return models

if __name__ == "__main__":
   loaded_models = ModelLoader.load_all_models("models")

