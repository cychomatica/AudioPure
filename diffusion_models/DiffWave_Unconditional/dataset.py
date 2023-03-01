## Modified based on https://github.com/pytorch/audio/blob/master/torchaudio/datasets/speechcommands.py

import os
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, models, transforms

from torch.utils.data.distributed import DistributedSampler
from scipy.io.wavfile import read as wavread

from typing import Tuple

import torchaudio
from torch.utils.data import Dataset
from torch import Tensor
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
)

HASH_DIVIDER = "_nohash_"
EXCEPT_FOLDER = "_background_noise_"

def fix_length(tensor, length):
    assert len(tensor.shape) == 2 and tensor.shape[0] == 1
    if tensor.shape[1] > length:
        return tensor[:,:length]
    elif tensor.shape[1] < length:
        return torch.cat([tensor, torch.zeros(1, length-tensor.shape[1])], dim=1)
    else:
        return tensor

#region
def load_speechcommands_item(filepath: str, path: str):
    relpath = os.path.relpath(filepath, path)
    label, filename = os.path.split(relpath)
    speaker, _ = os.path.splitext(filename)

    speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
    utterance_number = int(utterance_number)

    # Load audio
    waveform, sample_rate = torchaudio.load(filepath)
    return (fix_length(waveform, length=16000), sample_rate, label)

class SPEECHCOMMANDS(Dataset):
    """
    Create a Dataset for Speech Commands. Each item is a tuple of the form:
    waveform, sample_rate, label
    """

    def __init__(self, root: str, folder_in_archive: str):
        self._path = os.path.join(root, folder_in_archive)
        # walker = walk_files(self._path, suffix=".wav", prefix=True)
        walker = sorted(str(p) for p in Path(self._path).glob('*/*.wav'))
        walker = filter(lambda w: HASH_DIVIDER in w and EXCEPT_FOLDER not in w, walker)
        self._walker = list(walker)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
        fileid = self._walker[n]
        return load_speechcommands_item(fileid, self._path)

    def __len__(self) -> int:
        return len(self._walker)

def load_Speech_commands(path, batch_size=4, num_gpus=1):
    """
    Load speech commands dataset
    """
    Speech_commands_dataset = SPEECHCOMMANDS(root=path, folder_in_archive='')                                                       

    # distributed sampler
    train_sampler = DistributedSampler(Speech_commands_dataset) if num_gpus > 1 else None

    trainloader = torch.utils.data.DataLoader(Speech_commands_dataset, 
                                              batch_size=batch_size,  
                                              sampler=train_sampler,
                                              num_workers=4,
                                              pin_memory=False,
                                              drop_last=True)
    return trainloader
#endregion

#region
import librosa
QKWS_CLASSES = 'hey_android, hey_snapdragon, hi_galaxy, hi_lumina'.split(', ')

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

class QUALCOMMKEYWORD(Dataset):

    def __init__(self, root: str, 
                folder_in_archive: str='',
                usage: str='All', 
                transform=LoadAudio(), 
                classes=QKWS_CLASSES) -> None:
        super().__init__()

        all_classes = [d for d in classes if os.path.isdir(os.path.join(root, d)) and not d.startswith('_')]

        for c in classes:
            assert c in all_classes

        class_to_idx = {classes[i]: i for i in range(len(classes))}
        for c in all_classes:
            if c not in class_to_idx:
                class_to_idx[c] = len(classes) - 1
        
        data = []
        for c in all_classes:
            d = os.path.join(root, c)
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
        data = self.transform(data)
        return data

def set_audio_length(data, time):
    
    samples = data['samples']
    sample_rate = data['sample_rate']
    length = int(time * sample_rate)
    if length < len(samples):
        data['samples'] = samples[:length]
    elif length > len(samples):
        data['samples'] = np.pad(samples, (0, length - len(samples)), "constant")
    return data

def rand_audio_length_collate(batch):

    time = np.random.uniform(0.74751, 2.74751)
    batch = [set_audio_length(data, time) for data in batch]

    sample_rate = torch.tensor([data['sample_rate'] for data in batch])
    samples = torch.cat([torch.from_numpy(data['samples']).unsqueeze(0) for data in batch], dim=0)
    targets = torch.tensor([data['target'] for data in batch])

    return (samples.unsqueeze(1), sample_rate, targets)

def load_Qualcomm_keyword(path, batch_size=4, num_gpus=1):
    """
    Load Qualcomm keyword spotting dataset
    """
    Qualcomm_keyword_dataset = QUALCOMMKEYWORD(root=path, folder_in_archive='')                                                       

    # distributed sampler
    train_sampler = DistributedSampler(Qualcomm_keyword_dataset) if num_gpus > 1 else None

    trainloader = torch.utils.data.DataLoader(Qualcomm_keyword_dataset, 
                                              batch_size=batch_size,  
                                              sampler=train_sampler,
                                              collate_fn=rand_audio_length_collate, 
                                              num_workers=4,
                                              pin_memory=False,
                                              drop_last=True)
    return trainloader
#endregion
