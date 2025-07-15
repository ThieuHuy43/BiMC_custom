import os
import cv2
import numpy as np
import albumentations as A
from collections import defaultdict
from tqdm import tqdm

def count_samples(root_dir):
    """Count samples in each class"""
    class_samples = defaultdict(int)
    
    for class_dir in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_dir)
        if os.path.isdir(class_path):
            n_samples = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            class_samples[class_dir] = n_samples
    
    return class_samples

def create_augmentation_pipeline():
    """Create augmentation pipeline"""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.4),
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur(p=1),
        ], p=0.3),
        A.OneOf([
            A.HueSaturationValue(p=1),
            A.RGBShift(p=1),
        ], p=0.3),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5)
    ])

def augment_class(class_path, target_samples=200):
    """Augment images in a class directory"""
    # Get existing images
    images = [f for f in os.listdir(class_path) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current_samples = len(images)
    
    if current_samples >= target_samples:
        return 0
    
    # Calculate needed augmentations
    augmentations_needed = target_samples - current_samples
    transform = create_augmentation_pipeline()
    
    # Augment images
    aug_count = 0
    pbar = tqdm(total=augmentations_needed, desc=f"Augmenting {os.path.basename(class_path)}")
    
    while aug_count < augmentations_needed:
        # Select random image
        img_name = np.random.choice(images)
        img_path = os.path.join(class_path, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation
        augmented = transform(image=image)
        aug_image = augmented['image']
        
        # Save augmented image
        aug_name = f"aug_{aug_count}_{img_name}"
        aug_path = os.path.join(class_path, aug_name)
        cv2.imwrite(aug_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
        
        aug_count += 1
        pbar.update(1)
    
    pbar.close()
    return aug_count

def main():
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fruits_data')
    
    # Get initial sample counts
    print("Analyzing dataset...")
    class_samples = count_samples(root_dir)
    
    print("\nInitial class distribution:")
    for class_name, count in class_samples.items():
        print(f"{class_name}: {count} samples")
    
    # Augment classes with fewer than 200 samples
    print("\nStarting augmentation...")
    total_augmented = 0
    
    for class_name, count in class_samples.items():
        if count < 200:
            class_path = os.path.join(root_dir, class_name)
            augmented = augment_class(class_path)
            total_augmented += augmented
            print(f"Added {augmented} augmented images to {class_name}")
    
    print(f"\nTotal augmented images created: {total_augmented}")
    
    # Verify final distribution
    print("\nFinal class distribution:")
    final_samples = count_samples(root_dir)
    for class_name, count in final_samples.items():
        print(f"{class_name}: {count} samples")

if __name__ == "__main__":
    main()