import numpy as np
import torch
import nibabel as nb
import pandas as pd
import random
from torch.utils.data import Dataset

def random_flip_3d(image):
    if random.random() > 0.5:
        axis = random.choice([0, 1])
        image = np.flip(image, axis).copy()
    return image

class UKBDataset(Dataset):
    def __init__(self, json_path):
        # Load JSON and convert to list of records
        df = pd.read_json(json_path)
        self.t1_data = df.to_dict('records')

    def __len__(self):
        return len(self.t1_data)
  
    def __getitem__(self, index):
        item = self.t1_data[index]
        
        # Extract crop coordinates
        x1 = int(item['shape0'])
        x2 = int(item['shape1']) + 1
        y1 = int(item['shape2'])
        y2 = int(item['shape3']) + 1
        z1 = int(item['shape4'])
        z2 = int(item['shape5']) + 1
        
        # Load image
        img_path = item['high_path']
        try:
            T2high = nb.load(img_path).get_fdata().astype(np.float32)
        except:
            # Return zero tensor if loading fails
            return {'high': torch.zeros((1, 64, 64, 64))}

        # Normalize within foreground mask
        mask = T2high > 0
        mean_val = float(item['mean_high'])
        std_val = float(item['std_high'])
        
        if std_val < 1e-8:
            std_val = 1.0

        T2high[mask] = (T2high[mask] - mean_val) / std_val
        
        # Random shift values (not applied to cropping)
        shift_num_1 = random.choice([-3,-2,-1,0,1,2,3])
        shift_num_2 = random.choice([-3,-2,-1,0,1,2,3])
        shift_num_3 = random.choice([-3,-2,-1,0,1,2,3])
        
        T2high_block = T2high[x1:x2, y1:y2, z1:z2]
        
        # Data augmentation
        T2high_block = random_flip_3d(T2high_block)
        
        T2high_block = T2high_block.reshape((1,) + T2high_block.shape)
        T2high_block = torch.tensor(T2high_block, dtype=torch.float32)

        return {'high': T2high_block}
