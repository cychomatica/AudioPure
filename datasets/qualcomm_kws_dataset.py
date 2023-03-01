import os
import numpy as np
import librosa
from torch.utils.data import Dataset

CLASSES = 'hey_android, hey_snapdragon, hi_galaxy, hi_lumina'

class QualcommKeywordSpottingDataset(Dataset):

    def __init__(self, folder, 
                transform=None, classes=CLASSES) -> None:
        super().__init__()

        all_classes = [d for d in classes if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]

        for c in classes:
            assert c in all_classes

        class_to_idx = {classes[i]: i for i in range(len(classes))}
        for c in all_classes:
            if c not in class_to_idx:
                class_to_idx[c] = len(classes) - 1
        
        data = []
        for c in all_classes:
            d = os.path.join(folder, c)
            target = class_to_idx[c]
            for f in os.listdir(d):
                path = os.path.join(d, f)
                data.append((path, target))
        
        self.classes = classes
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        path, target = self.data[index]
        data = {'path': path, 'target': target}

        if self.transform is not None:
            data = self.transform(data)

        return data
