import librosa.display
import matplotlib.pyplot as plt
from typing import Union
import numpy as np
import torch
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
        path = './_spec_samples'
    if not os.path.exists(path):
        os.makedirs(path)
    if name is None:
        name = 'spec.png'
    fig.savefig(os.path.join(path, name))
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 10), sharey=True)
# p1 = librosa.display.specshow(batch[0].squeeze().numpy(), 
#                                 x_axis='ms', y_axis='mel', 
#                                 sr=16000, n_fft=2048, 
#                                 fmin=0, fmax=8000, 
#                                 ax=ax1, cmap='magma')
# p2 = librosa.display.specshow(batch_purified[0].detach().cpu().squeeze().numpy(), 
#                                 x_axis='ms', y_axis='mel', 
#                                 sr=16000, n_fft=2048, 
#                                 fmin=0, fmax=8000, 
#                                 ax=ax2, cmap='magma')
# # p2 = librosa.display.specshow(batch_purified[0].squeeze().detach().cpu().numpy(), ax=ax2, y_axis='log', x_axis='time')
# plt.tight_layout()
# fig.colorbar(p1, ax=ax1, format="%+2.f dB")
# fig.colorbar(p2, ax=ax2, format="%+2.f dB")
# fig.savefig('spec.png')