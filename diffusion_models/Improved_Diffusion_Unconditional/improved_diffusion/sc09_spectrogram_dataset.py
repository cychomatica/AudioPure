"""Google speech commands dataset."""
__author__ = 'Yuan Xu'

import os
import numpy as np

import librosa

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import *
import torchaudio
import torch

# n_mels = 32
# MelSpecTrans = torchaudio.transforms.MelSpectrogram(n_fft=2048, hop_length=512, n_mels=n_mels, norm='slaney', pad_mode='constant', mel_scale='slaney')
# Amp2DB = torchaudio.transforms.AmplitudeToDB(stype='power')
# Wave2Spect = Compose([MelSpecTrans.cuda(), Amp2DB.cuda()])

__all__ = [ 'CLASSES', 'SpeechCommandsDataset', 'SC09_Spectrogram_Dataset', 'BackgroundNoiseDataset' ]

'''sc'''
CLASSES = 'unknown, silence, yes, no, up, down, left, right, on, off, stop, go'.split(', ')

'''sc09'''
SC09_CLASSES = 'zero, one, two, three, four, five, six, seven, eight, nine'.split(', ')

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

# calculated from full SC09 Mle-spectrogram dataset
MEL_UPPER_BOUND = 38.22
MEL_LOWER_BOUND = -100.0

def melspec_standardize(x):
    '''
        scale mel-spectrogram to [-1,1]
    '''
    # assert x.max() <= MEL_UPPER_BOUND and x.min() >= MEL_LOWER_BOUND
    x = 2 * (x - MEL_LOWER_BOUND) / (MEL_UPPER_BOUND - MEL_LOWER_BOUND) - 1
    # assert x.max() <= 1 and x.min() >= -1
    return x

def melspec_inv_standardize(x):
    '''
        map values in [-1,1] back to mel-scale
    '''
    # assert x.max() <= 1 and x.min() >= -1; ''
    x = (x + 1) * (MEL_UPPER_BOUND - MEL_LOWER_BOUND) / 2 + MEL_LOWER_BOUND
    # assert x.max() <= MEL_UPPER_BOUND and x.min() >= MEL_LOWER_BOUND
    return x

def load_sc09_data(data_dir, batch_size, n_mels=32, class_cond=False, deterministic=False):

    WaveTrans = Compose([LoadAudio(), FixAudioLength()])
    MelSpecTrans = torchaudio.transforms.MelSpectrogram(n_fft=2048, hop_length=512, n_mels=n_mels, norm='slaney', pad_mode='constant')  # for low version torchaudio
    # MelSpecTrans = torchaudio.transforms.MelSpectrogram(n_fft=2048, hop_length=512, n_mels=n_mels, norm='slaney', pad_mode='constant', mel_scale='slaney')
    Amp2DB = torchaudio.transforms.AmplitudeToDB(stype='power')
    Wave2Spect = Compose([MelSpecTrans, Amp2DB]) # waveform (batch_size, 1, length) -> spectrogram (batch_size, 1, n_mels, 32)

    dataset = SC09_Spectrogram_Dataset(folder=data_dir, wave_trans=WaveTrans, class_cond=class_cond, wave_to_spect=Wave2Spect)

    if deterministic:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    while True:
        yield from loader
    # return loader

class SpeechCommandsDataset(Dataset):
    """Google speech commands dataset. Only 'yes', 'no', 'up', 'down', 'left',
    'right', 'on', 'off', 'stop' and 'go' are treated as known classes.
    All other classes are used as 'unknown' samples.
    See for more information: https://www.kaggle.com/c/tensorflow-speech-recognition-challenge
    """

    def __init__(self, folder, transform=None, classes=SC09_CLASSES, silence_percentage=0.1):

        # all_classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]

        all_classes = [d for d in classes if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]

        for c in classes[2:]:
           assert c in all_classes

        class_to_idx = {classes[i]: i for i in range(len(classes))}
        for c in all_classes:
            if c not in class_to_idx:
                # class_to_idx[c] = 0
                class_to_idx[c] = len(classes) - 1

        data = []
        for c in all_classes:
            d = os.path.join(folder, c)
            target = class_to_idx[c]
            for f in os.listdir(d):
                path = os.path.join(d, f)
                data.append((path, target))

        # add silence
        # target = class_to_idx['silence']
        # data += [('', target)] * int(len(data) * silence_percentage)

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

    def make_weights_for_balanced_classes(self):
        """adopted from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3"""

        nclasses = len(self.classes)
        count = np.zeros(nclasses)
        for item in self.data:
            count[item[1]] += 1

        N = float(sum(count))
        weight_per_class = N / count
        weight = np.zeros(len(self))
        for idx, item in enumerate(self.data):
            weight[idx] = weight_per_class[item[1]]
        return weight

class SC09_Spectrogram_Dataset(Dataset):
    """Google speech commands 0~9 subset. Only 'zero', 'one', 'two', 'three', 'four',
    'five', 'six', 'seven', 'eight' and 'nine' are included.
    All other classes are excluded.
    See for more information: https://www.kaggle.com/c/tensorflow-speech-recognition-challenge
    """

    def __init__(self, folder, wave_trans=None, wave_to_spect=None, classes=SC09_CLASSES, class_cond=False, num_per_class=74751):


        all_classes = [d for d in classes if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]
        
        for c in classes:
           assert c in all_classes

        class_to_idx = {classes[i]: i for i in range(len(classes))}
        for c in all_classes:
            if c not in class_to_idx:
                # class_to_idx[c] = 0
                class_to_idx[c] = len(classes) - 1

        data = []
        for c in all_classes:
            d = os.path.join(folder, c)
            target = class_to_idx[c]
            for f in os.listdir(d)[:min(num_per_class, len(os.listdir(d)))]:
                path = os.path.join(d, f)
                data.append((path, target))

        self.used_classes = classes if class_cond else None
        self.data = data
        self.wave_trans = wave_trans
        self.wave_to_spect = wave_to_spect

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, target = self.data[index]
        data = {'path': path, 'target': target}

        wave_data = self.wave_trans(data)
        waveform, target = torch.from_numpy(wave_data['samples']).unsqueeze(0), wave_data['target']
        spectrogram = self.wave_to_spect(waveform)

        out_dict = {}
        if self.used_classes is not None:
            out_dict['y'] = np.array(target, dtype=np.int64)
        
        return spectrogram.numpy(), out_dict

    def make_weights_for_balanced_classes(self):
        """adopted from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3"""

        nclasses = len(self.used_classes)
        count = np.zeros(nclasses)
        for item in self.data:
            count[item[1]] += 1

        N = float(sum(count))
        weight_per_class = N / count
        weight = np.zeros(len(self))
        for idx, item in enumerate(self.data):
            weight[idx] = weight_per_class[item[1]]
        return weight

class BackgroundNoiseDataset(Dataset):
    """Dataset for silence / background noise."""

    def __init__(self, folder, transform=None, sample_rate=16000, sample_length=1):
        audio_files = [d for d in os.listdir(folder) if os.path.isfile(os.path.join(folder, d)) and d.endswith('.wav')]
        samples = []
        for f in audio_files:
            path = os.path.join(folder, f)
            s, sr = librosa.load(path, sr=sample_rate)
            samples.append(s)

        samples = np.hstack(samples)
        c = int(sample_rate * sample_length)
        r = len(samples) // c
        self.samples = samples[:r*c].reshape(-1, c)
        self.sample_rate = sample_rate
        self.classes = CLASSES
        self.transform = transform
        self.path = folder

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = {'samples': self.samples[index], 'sample_rate': self.sample_rate, 'target': 1, 'path': self.path}

        if self.transform is not None:
            data = self.transform(data)

        return data
