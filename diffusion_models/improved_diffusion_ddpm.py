from random import gauss
import torch
import numpy as np
from typing import Union

from .Improved_Diffusion_Unconditional.improved_diffusion.script_util import create_model_and_diffusion
from .Improved_Diffusion_Unconditional.improved_diffusion.gaussian_diffusion import GaussianDiffusion
from .Improved_Diffusion_Unconditional.improved_diffusion.unet import UNetModel
from .Improved_Diffusion_Unconditional.improved_diffusion.sc09_spectrogram_dataset import melspec_standardize, melspec_inv_standardize

class ImprovedDiffusion(torch.nn.Module):
    
    def __init__(self,
                 model: UNetModel=None, 
                 diffusion: GaussianDiffusion=None, 
                 reverse_timestep: int=0
                 ):
        super().__init__()
        
        '''
            unconditional improved diffusion for adversarial purification
        '''
        
        self.model = model
        self.diffusion = diffusion
        self.reverse_timestep = reverse_timestep
        
    def forward(self, waveforms: Union[torch.Tensor, np.ndarray]):
        
        if isinstance(waveforms, np.ndarray): 
            waveforms = torch.from_numpy(waveforms)
        
        # with torch.no_grad():
        output = self._diffusion(waveforms)
        output = self._reverse(output)
        
        '''
            the original mel-spectrogram is divided by 100 to get x_0 ranging in [-1,1]; 
            the output should multiply 100 to remap to mel-scle
        '''
        output = melspec_inv_standardize(output)
        
        return output

    def _diffusion(self, x_0: Union[torch.Tensor, np.ndarray]) -> torch.Tensor: 
        
        if isinstance(x_0, np.ndarray): 
            x_0 = torch.from_numpy(x_0)
            
        x_t = self.diffusion.q_sample(x_0, t=self.reverse_timestep)
        return x_t
            
    def _reverse(self, x_t: Union[torch.Tensor, np.ndarray]) -> torch.Tensor: 

        '''convert np.array to torch.tensor'''
        if isinstance(x_t, np.ndarray): 
            x_t = torch.from_numpy(x_t)
        
        self.diffusion.p_sample_loop(model=self.model, shape=x_t.shape)

import argparse
from .Improved_Diffusion_Unconditional.improved_diffusion import dist_util
from .Improved_Diffusion_Unconditional.improved_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser, args_to_dict

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    
    defaults['image_size'] = 32
    defaults['num_channels'] = 128
    defaults['num_res_blocks'] = 3
    defaults['learn_sigma'] = False
    defaults['diffusion_steps'] = 1000
    defaults['noise_schedule'] = 'linear'
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def create_improved_diffusion(model_path, reverse_timestep=25):
    
    args = create_argparser().parse_args()
    
    UNet_model, gaussian_diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    UNet_model.load_state_dict(dist_util.load_state_dict(model_path, map_location="cpu"))
    Purifier = ImprovedDiffusion(model=UNet_model, 
                                 diffusion=gaussian_diffusion, 
                                 reverse_timestep=reverse_timestep)
    
    return Purifier
    