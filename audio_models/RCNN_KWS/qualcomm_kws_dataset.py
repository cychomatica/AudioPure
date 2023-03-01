import os
import numpy as np
import librosa
from torch.utils.data import Dataset
import torch

CLASSES = 'hey_android, hey_snapdragon, hi_galaxy, hi_lumina'.split(', ')

class QualcommKeywordSpottingDataset(Dataset):

    def __init__(self, folder, usage='Train', 
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

            data_c = []
            for root, dirs, files in os.walk(d, topdown=False):
                for name in files:
                    path = os.path.join(root, name)
                    if path.endswith('.wav'):
                        data_c.append((path, target))

            '''if usage is not in {'Train', 'Valid', 'Test'}, use the whole dataset'''
            if usage == 'Train':
                data_c = data_c[:-125]
            elif usage == 'Valid':
                data_c = data_c[-125:-25]
            elif usage == 'Test':
                data_c = data_c[-25:]

            data.extend(data_c)

        
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

class LoadAudio(object):
    """Loads an audio into a numpy array."""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def __call__(self, data):
        path = data['path']
        if path:
            samples, sample_rate = librosa.load(path, sr=self.sample_rate)
        else:
            # silence
            sample_rate = self.sample_rate
            samples = np.zeros(sample_rate, dtype=np.float32)
        data['samples'] = samples
        data['sample_rate'] = sample_rate
        return data

class FixAudioLength(object):
    """Either pads or truncates an audio into a fixed length."""

    def __init__(self, time=1):
        self.time = time

    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        length = int(self.time * sample_rate)
        if length < len(samples):
            data['samples'] = samples[:length]
        elif length > len(samples):
            data['samples'] = np.pad(samples, (0, length - len(samples)), "constant")
        return data

class RandAudioLength(object):
    """Either pads or truncates an audio into a fixed length."""

    def __init__(self, time_min=1, time_max=2):
        self.time_min = time_min
        self.time_max = time_max

    def __call__(self, data):

        time = np.random.uniform(self.time_min, self.time_max)
        samples = data['samples']
        sample_rate = data['sample_rate']
        length = int(time * sample_rate)
        if length < len(samples):
            data['samples'] = samples[:length]
        elif length > len(samples):
            data['samples'] = np.pad(samples, (0, length - len(samples)), "constant")
        return data
