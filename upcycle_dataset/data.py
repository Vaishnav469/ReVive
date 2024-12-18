import os
import random
from shutil import copy2
from PIL import Image
from torchvision import transforms

# Paths to your dataset
dataset_dir = "./"
output_dir = "./"
target_size = 500  # Target number of images per class

# Labels to combine
combine_labels = {
    "plastic_bottles": ["plastic_soda_bottles", "plastic_water_bottles"],
    "glass_bottles": ["glass_beverage_bottles", "glass_food_jars"]
}

# Data augmentation transformations
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
])

def create_combined_labels_with_unique_names(dataset_dir, output_dir, combine_labels, images_per_label=250):
    """Combine specific number of images from specified labels with unique filenames."""
    os.makedirs(output_dir, exist_ok=True)
    
    for new_label, old_labels in combine_labels.items():
        combined_path = os.path.join(output_dir, new_label)
        os.makedirs(combined_path, exist_ok=True)
        
        for old_label in old_labels:
            old_path = os.path.join(dataset_dir, old_label)
            images = os.listdir(old_path)
            random.shuffle(images)  # Shuffle images
            selected_images = images[:images_per_label]  # Take the specified number of images
            
            for img_name in selected_images:
                img_path = os.path.join(old_path, img_name)
                # Create a unique name by prefixing the old label
                new_img_name = f"{old_label}_{img_name}"
                new_img_path = os.path.join(combined_path, new_img_name)
                
                # Copy the image to the combined folder
                copy2(img_path, new_img_path)
        
        print(f"Combined {new_label} created with {len(os.listdir(combined_path))} images.")



def augment_images(label_dir, target_size, transforms):
    """Augment images to reach the target size."""
    images = [os.path.join(label_dir, img) for img in os.listdir(label_dir)]
    count = len(images)
    
    if count >= target_size:
        return  # No augmentation needed

    print(f"Augmenting {label_dir}: {count} -> {target_size}")
    while len(images) < target_size:
        # Randomly select an image for augmentation
        img_path = random.choice(images)
        try:
            img = Image.open(img_path)
            
            # Convert RGBA to RGB if needed
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            img_augmented = transforms(img)
            
            # Save augmented image
            new_img_name = f"aug_{len(images)}_{os.path.basename(img_path)}"
            img_augmented.save(os.path.join(label_dir, new_img_name), format='JPEG')
            images.append(os.path.join(label_dir, new_img_name))
        except (UnidentifiedImageError, IsADirectoryError) as e:
            print(f"Skipped file: {img_path}, Reason: {e}")

# Main workflow
if __name__ == "__main__":
    create_combined_labels_with_unique_names(dataset_dir, output_dir, combine_labels, images_per_label=250)
  
    # Step 2: Augment combined labels to the target size
    for label in ['aerosol_cans', 'newspaper', 'aluminum_soda_cans', 'cardboard_boxes', 'hoodies', 'polo_shirt', 'shirt', 'tank_top']:
        label_dir = os.path.join(output_dir, label)
        augment_images(label_dir, target_size, augmentation_transforms)
    
    print("Dataset preparation complete!")
