import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(source_dir, train_dir, test_dir, test_size=0.2, random_state=42):
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=random_state)
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    for file in train_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, file))
    
    for file in test_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(test_dir, file))
    
    print(f"Dataset split complete. {len(train_files)} training images, {len(test_files)} test images.")

if __name__ == "__main__":
    source_dir = "./all-dogs"
    train_dir = "./data/train/dogs"
    test_dir = "./data/test/dogs"
    
    split_dataset(source_dir, train_dir, test_dir)