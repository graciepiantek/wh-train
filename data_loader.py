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

    def _load_data_with_temporal_split(self, data_path):
        class_dirs = self._detect_dataset_structure(data_path)
        
        all_train_files = []
        all_val_files = []
        
        for class_name, class_path in class_dirs.items():
            print(f"Processing {class_name} from {class_path}")
            
            class_files = []
            for f in os.listdir(class_path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    class_files.append((os.path.join(class_path, f), class_name))
            
            print(f"  Found {len(class_files)} files in {class_name}")
            
            train_files, val_files = self._temporal_split_files_balanced(class_files)
            all_train_files.extend(train_files)
            all_val_files.extend(val_files)
        
        train_generator = self._create_generator_from_files(all_train_files, augment=True)
        val_generator = self._create_generator_from_files(all_val_files, augment=False)
        
        print(f"\n Total split: {len(all_train_files)} train, {len(all_val_files)} val files")
        
        return train_generator, val_generator

    def _detect_dataset_structure(self, data_path):
        contents = os.listdir(data_path)
    
        if 'signal' in contents and 'no_signal' in contents:
            return {
                'signal': os.path.join(data_path, 'signal'),
                'no_signal': os.path.join(data_path, 'no_signal')
            }
        
        elif 'train' in contents and 'no_train' in contents:
            return {
                'train': os.path.join(data_path, 'train'),
                'no_train': os.path.join(data_path, 'no_train')
            }
        
        elif 'hayden' in contents:
            hayden_path = os.path.join(data_path, 'hayden')
            hayden_contents = os.listdir(hayden_path)
            
            if 'train' in hayden_contents and 'no_train' in hayden_contents:
                return {
                    'train': os.path.join(hayden_path, 'train'),
                    'no_train': os.path.join(hayden_path, 'no_train')
                }
        
        else:
            raise ValueError(f"Unknown dataset structure in {data_path}. Contents: {contents}")

    def _extract_date_from_filename(self, filename):
        print(f"DEBUG: Processing filename: {filename}")

        prefix_match = re.search(r'^([a-z]+)-(\d{8})', filename)
        if prefix_match:
            prefix = prefix_match.group(1)
            date = prefix_match.group(2)
            
            if prefix in ['da', 'dp']: 
                result = f'daytime_{date}'
            elif prefix in ['na', 'np']:
                result = f'nighttime_{date}'
            else:
                result = f'{prefix}_{date}'
            
            print(f"DEBUG: Prefix → {result}")
            return result

        if filename.startswith('2020') and len(filename) >= 15:
            date = filename[:8]  
            time_part = filename[9:15]
            hour = int(time_part[:2])
            
            if 6 <= hour <= 18:
                result = f'daytime_{date}'
            else:
                result = f'nighttime_{date}'
            
            print(f"DEBUG: Timestamp → {result}")
            return result

        if filename.startswith('other_cam_'):
            cam_match = re.search(r'other_cam_([^_]+)', filename)
            if cam_match:
                cam_type = cam_match.group(1)
                result = f'other_cam_night_{cam_type}' if 'nt' in cam_type else f'other_cam_day_{cam_type}'
                print(f"DEBUG: Camera → {result}")
                return result
        
        print(f"DEBUG: No match → unknown")
        return 'unknown'

    def _temporal_split_files_balanced(self, file_class_pairs, val_split=0.15, seed=42):
        timestamp_groups = defaultdict(list)
        camera_groups = defaultdict(list)
        unknown_files = []
        
        for file_path, class_name in file_class_pairs:
            filename = os.path.basename(file_path)
            date_key = self._extract_date_from_filename(filename)
            
            if date_key == 'unknown':
                unknown_files.append((file_path, class_name))
            elif date_key.startswith('other_cam_'):
                camera_groups[date_key].append((file_path, class_name))
            elif date_key.isdigit() and len(date_key) == 8:
                timestamp_groups[date_key].append((file_path, class_name))
        
        print(f"Found {len(timestamp_groups)} timestamp groups, {len(camera_groups)} camera groups")
        
        train_files = []
        val_files = []
        
        total_files = len(file_class_pairs)
        total_val_target = int(total_files * val_split)
        
        all_groups = {**timestamp_groups, **camera_groups}
        
        if len(all_groups) >= 2:
            files_per_group_val = total_val_target // len(all_groups)
            remainder = total_val_target % len(all_groups)
            
            print(f"Target: {files_per_group_val} val files per group (+{remainder} extra)")
            
            group_val_counts = {}
            
            for i, (group_name, group_files) in enumerate(sorted(all_groups.items())):
                val_count = files_per_group_val + (1 if i < remainder else 0)
                val_count = min(val_count, len(group_files))
            
                if val_count > 0:
                    np.random.seed(seed + i)
                    indices = np.random.choice(len(group_files), size=val_count, replace=False)
                    group_val_files = [group_files[idx] for idx in indices]
                    group_train_files = [group_files[idx] for idx in range(len(group_files)) if idx not in indices]
                    
                    val_files.extend(group_val_files)
                    train_files.extend(group_train_files)
                    group_val_counts[group_name] = val_count
                    
                    print(f"{group_name}: {len(group_train_files)} train, {val_count} val")
                else:
                    train_files.extend(group_files)
                    group_val_counts[group_name] = 0
                    print(f"      {group_name}: {len(group_files)} train, 0 val")
            
            print(f"Balanced validation distribution: {group_val_counts}")
            
        else:
            print("Insufficient groups for balanced split, using random split")
            if all_groups:
                all_files = [f for files in all_groups.values() for f in files]
                train_files, val_files = train_test_split(all_files, test_size=val_split, random_state=seed)
            else:
                train_files = file_class_pairs
                val_files = []
        
        if unknown_files:
            print(f"Adding {len(unknown_files)} unknown files to training")
            train_files.extend(unknown_files)
        
        print(f"Final split: {len(train_files)} train, {len(val_files)} val")
        
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