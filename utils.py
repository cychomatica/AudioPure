import torch
import torchaudio
import numpy as np

import librosa.display
import matplotlib.pyplot as plt
from typing import Union
import os

def spec_save(x: Union[np.ndarray, torch.Tensor], path=None, name=None):
    
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = x.squeeze()
    assert x.shape == (32, 32)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(data=x, 
                                   x_axis='ms', y_axis='mel', 
                                   sr=16000, n_fft=2048, 
                                   fmin=0, fmax=8000, 
                                   ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    
    if path is None:
        path = './_Spec_Samples'
    if not os.path.exists(path):
        os.makedirs(path)
    if name is None:
        name = 'spec.png'
    fig.savefig(os.path.join(path, name))

def audio_save(x: Union[np.ndarray, torch.Tensor], path=None, name=None):

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.detach().cpu()
    assert x.ndim == 2 and x.shape[0] == 1

    if path is None:
        path = './_Audio_Samples'
    if not os.path.exists(path):
        os.makedirs(path)
    if name is None:
        name = 'audio.wav'

    torchaudio.save(os.path.join(path,name), x, 16000) # default sample rate = 16000

def audio_save_as_img(x: Union[np.ndarray, torch.Tensor], path=None, name=None, color=None):
    
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = x.squeeze()
    assert x.ndim == 1

    fig = plt.figure(figsize=(21, 9), dpi=100)

    from scipy.interpolate import make_interp_spline

    # x_smooth = make_interp_spline(np.arange(0, len(x)), x)(np.linspace(0, len(x), 1000))
    # plt.ylim(-1.5*max(abs(x.max()), abs(x.min())),1.5*max(abs(x.max()), abs(x.min())))
    # plt.plot((np.linspace(0, len(x), 1000)), x_smooth,'-')
    # plt.ylim(-1,1)
    plt.plot(x,'-',color=color if color is not None else 'steelblue', transparent=True)

    if path is None:
        path = './_Audio_Samples'
    if not os.path.exists(path):
        os.makedirs(path)
    if name is None:
        name = 'waveform.png'

    fig.savefig(os.path.join(path, name))
