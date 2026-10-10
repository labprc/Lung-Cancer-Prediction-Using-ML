import os
import shutil
import random
from glob import glob

def main():
    src_dir = r"C:\Users\vedan\OneDrive\Desktop\deases\lung-cancer-ct-ai\data\train"
    base_dir = r"C:\Users\vedan\OneDrive\Desktop\deases\lung-cancer-ct-ai\data\split"

    dirs = {'train': os.path.join(base_dir, 'train'),
            'val': os.path.join(base_dir, 'val'),
            'test': os.path.join(base_dir, 'test')}

    for d in dirs.values():
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    
    total_images = 0
    splits_count = {'train': 0, 'val': 0, 'test': 0}
    class_dist = {
        'train': {c: 0 for c in classes},
        'val': {c: 0 for c in classes},
        'test': {c: 0 for c in classes}
    }

    random.seed(42)

    for cls in classes:
        for d in dirs.values():
            os.makedirs(os.path.join(d, cls), exist_ok=True)
            
        images = glob(os.path.join(src_dir, cls, "*.*"))
        random.shuffle(images)
        
        # Test = 15%, Val = 15%, Train = 70%
        # To strictly maintain the existing 183 test size out of 1226, 15% is perfect.
        test_size = int(len(images) * 0.15)
        val_size = int(len(images) * 0.15)
        
        test_imgs = images[:test_size]
        val_imgs = images[test_size:test_size+val_size]
        train_imgs = images[test_size+val_size:]
        
        total_images += len(images)
        
        for img in train_imgs:
            shutil.copy(img, os.path.join(dirs['train'], cls, os.path.basename(img)))
            splits_count['train'] += 1
            class_dist['train'][cls] += 1
            
        for img in val_imgs:
            shutil.copy(img, os.path.join(dirs['val'], cls, os.path.basename(img)))
            splits_count['val'] += 1
            class_dist['val'][cls] += 1
            
        for img in test_imgs:
            shutil.copy(img, os.path.join(dirs['test'], cls, os.path.basename(img)))
            splits_count['test'] += 1
            class_dist['test'][cls] += 1

    print(f"TOTAL DATASET:\n{total_images} images\n")
    print(f"TRAIN:\n{splits_count['train']}")
    print(f"VALIDATION:\n{splits_count['val']}")
    print(f"TEST:\n{splits_count['test']}\n")
    
    for split in ['train', 'val', 'test']:
        print(f"{split.upper()}:")
        for cls in classes:
            print(f"{cls}: {class_dist[split][cls]}")
        print()
        
if __name__ == "__main__":
    main()
