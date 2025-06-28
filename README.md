## Project Set-up
```mermaid
graph TD
    A[wh-train] --> B[configs/]
    A --> C[models/]
    A --> D[results/]
    A --> E[Scripts]
    
    B --> B1[chestnut_signal_config.json]
    B --> B2[fourth_signal_config.json]
    B --> B3[train_config.json]
    
    C --> C1[original/]
    C --> C2[chestnut_original/]
    C --> C3[fourth_original/]
    
    C1 --> C1a[chestnut_signal/]
    C1 --> C1b[chestnut_train/]
    C1 --> C1c[fourth_signal/]
    C1 --> C1d[fourth_train/]
    
    D --> D1[test/]
    D --> D2[test_original/]
    D --> D3[test_resnet/]
    
    E --> E1[main.py]
    E --> E2[data_loader.py]
    E --> E3[evaluator.py]
    E --> E4[model_loader.py]
    E --> E5[analyze_results.py]
    E --> E6[run_all_evaluations.sh]

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