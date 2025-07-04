## Project Structure


## Models Tested

### 1. Original Models (.keras/.hdf5)
- Pre-trained custom CNN architectures
- Optimized specifically for railway intersection monitoring

### 2. EfficientNetV2S (Transfer Learning)
- Pre-trained on ImageNet
- Fine-tuned with GlobalAveragePooling2D + Dense layers

### 3. ResNet152V2 (Transfer Learning)  
- Pre-trained on ImageNet

## Key Findings


## Technical Setup


### Configuration Files
- `configs/chestnut_signal_config.json`
- `configs/fourth_signal_config.json`
- `configs/train_config.json` 

### Running Evaluations



## Contributors

* Gracie Piantek - Transfer learning implementation and evaluation
* Hayden Roche - Original train spotter models and dataset

## References

Original train spotter project: https://github.com/haydenroche5/train_spotter