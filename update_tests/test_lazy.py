
import torch
from torch.utils.data import DataLoader
from dataset import LazyTranslationDataset, collate_fn
import os
import time
import shutil

# Create dummy large file
def create_dummy_data():
    os.makedirs('temp_data', exist_ok=True)
    with open('temp_data/train.en', 'w') as f_src, open('temp_data/train.de', 'w') as f_tgt:
        for i in range(100000):
            f_src.write(f"This is source sentence number {i} with some random tokens.\n")
            f_tgt.write(f"Das ist Ziel Satz Nummer {i} mit einigen zufälligen Token.\n")

def test_lazy_dataset():
    print("Creating dummy data...")
    create_dummy_data()
    
    print("Initializing LazyTranslationDataset...")
    
    dataset = LazyTranslationDataset(
        'temp_data/train.en',
        'temp_data/train.de'
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test DataLoader with multiple workers
    print("Testing DataLoader with num_workers=4...")
    loader = DataLoader(dataset, batch_size=32, num_workers=4, collate_fn=collate_fn)
    
    for i, batch in enumerate(loader):
        if i >= 10: break
        pass
        
    print("DataLoader iteration successful.")
    
    # Clean up
    shutil.rmtree('temp_data')
    print("Test Passed.")

if __name__ == "__main__":
    test_lazy_dataset()
