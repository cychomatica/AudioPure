import argparse
import json

import torch
from .DiffWave_Unconditional.dataset import load_Qualcomm_keyword
from .DiffWave_Unconditional.WaveNet import WaveNet_Speech_Commands
from .DiffWave_Unconditional.util import calc_diffusion_hyperparams

import numpy as np
from typing import Union

# import sys
# sys.path.insert(0, './')
# import utils

class DiffWave(torch.nn.Module):

    def __init__(self, 
                model: WaveNet_Speech_Commands, 
                diffusion_hyperparams: dict,
                reverse_timestep: int=200, 
                grad_enable=True
                ):
        super().__init__()

        '''
            model: input (x_t, t), output epsilon_theta at timestep t
        '''

        self.model = model
        self.diffusion_hyperparams = diffusion_hyperparams
        self.reverse_timestep = reverse_timestep
        self.freeze = False
        self.grad_enable = grad_enable

    def forward(self, waveforms: Union[torch.Tensor, np.ndarray]):

        if isinstance(waveforms, np.ndarray): 
            waveforms = torch.from_numpy(waveforms)
        
        with torch.no_grad():
            output = self._diffusion(waveforms)
            output = self._reverse(output)
        # output = self.one_shot_denoise(output)
        # output = self.fast_reverse(output)

        return output

    def _diffusion(self, x_0: Union[torch.Tensor, np.ndarray]) -> torch.Tensor: 
        
        '''convert np.array to torch.tensor'''
        if isinstance(x_0, np.ndarray): 
            x_0 = torch.from_numpy(x_0)

        T, Alpha, Alpha_bar, Sigma = self.diffusion_hyperparams["T"], \
                                    self.diffusion_hyperparams["Alpha"], \
                                    self.diffusion_hyperparams["Alpha_bar"], \
                                    self.diffusion_hyperparams["Sigma"] 
        assert len(Alpha) == T
        assert len(Alpha_bar) == T
        assert len(Sigma) == T
        assert x_0.ndim == 3

        '''noising'''
        # with torch.no_grad():
        z = torch.normal(0, 1, size=x_0.shape).cuda()
        x_t = torch.sqrt(Alpha_bar[self.reverse_timestep-1]).cuda() * x_0 + torch.sqrt(1-Alpha_bar[self.reverse_timestep-1]).cuda() * z
            # x_t = x_0
            # for t in range(0, self.reverse_timestep):
            #     z = torch.normal(0, 1, size=x_0.shape).cuda()
            #     x_t = torch.sqrt(Alpha[t]).cuda() * x_0 + torch.sqrt(1-Alpha[t]).cuda() * z
            #     utils.audio_save_as_img(x_t[0], name='wave_adv_diff_{}.png'.format(t+1))
        return x_t

    def _reverse(self, x_t: Union[torch.Tensor, np.ndarray]) -> torch.Tensor: 

        '''convert np.array to torch.tensor'''
        if isinstance(x_t, np.ndarray): 
            x_t = torch.from_numpy(x_t)

        T, Alpha, Alpha_bar, Sigma = self.diffusion_hyperparams["T"], \
                                    self.diffusion_hyperparams["Alpha"], \
                                    self.diffusion_hyperparams["Alpha_bar"], \
                                    self.diffusion_hyperparams["Sigma"] 
        assert len(Alpha) == T
        assert len(Alpha_bar) == T
        assert len(Sigma) == T
        assert x_t.ndim == 3

        '''denoising'''
        x_t_rev = x_t.clone()
        # with torch.no_grad():

        # compute x_t-1 from x_t
        for t in range(self.reverse_timestep-1, -1, -1):

            epsilon_theta_t, mu_theta_t, sigma_thata_t = self.compute_coefficients(x_t_rev, t)

            if t > 0:
                x_t_rev = mu_theta_t + sigma_thata_t * torch.normal(0, 1, size=x_t_rev.shape).cuda()
            else:
                x_t_rev = mu_theta_t
                # utils.audio_save_as_img(x_t_rev[0], name='wave_adv_rev_{}.png'.format(t))
        return x_t_rev
    
    def fast_reverse(self, x_t: Union[torch.Tensor, np.ndarray]) -> torch.Tensor: 

        '''convert np.array to torch.tensor'''
        if isinstance(x_t, np.ndarray): 
            x_t = torch.from_numpy(x_t)

        T, Alpha, Alpha_bar, Sigma = self.diffusion_hyperparams["T"], \
                                    self.diffusion_hyperparams["Alpha"], \
                                    self.diffusion_hyperparams["Alpha_bar"], \
                                    self.diffusion_hyperparams["Sigma"]

        K = 3
        S = torch.linspace(1, self.reverse_timestep, K)
        S = torch.round(S).int() - 1
        Beta_new, Beta_tilde_new = torch.zeros(size=(K,)), torch.zeros(size=(K,))

        for i in range(K):
            if i > 0:
                Beta_new[i] =  1 - Alpha_bar[S[i]] / Alpha_bar[S[i-1]]
                Beta_tilde_new[i] = (1 - Alpha_bar[S[i-1]]) / (1 - Alpha_bar[S[i]]) * Beta_new[i]
            else:
                Beta_new[i] =  1 - Alpha_bar[S[i]]
                Beta_tilde_new[i] = 0
        Alpha_new = 1 - Beta_new
        Alpha_bar_new = torch.cumprod(Alpha_new, dim=0)

        x_St = x_t
        for t in range(K-1, -1, -1):

            real_t = S[t]
            eps_St = self.model((x_St, real_t * torch.ones((x_St.shape[0], 1)).cuda()))
            mu_St = (x_St - (1 - Alpha_new[t]) / torch.sqrt(1 - Alpha_bar_new[t]) * eps_St) / torch.sqrt(Alpha_new[t])
            sigma_St = Beta_tilde_new[t]
            x_St = mu_St + sigma_St * torch.normal(0, 1, size=x_St.shape).cuda()

        return x_St

    def compute_coefficients(self, x_t: Union[torch.Tensor, np.ndarray], t: int):

        '''
            a single reverse step
            compute coefficients at timestep t+1
            t: in [0, T-1]
            return: eps_theta(x_t+1, t+1), mu_theta(x_t+1, t+1) and sigma_theta(x_t+1, t+1)
        '''

        Alpha, Alpha_bar, Sigma = self.diffusion_hyperparams["Alpha"], \
                                self.diffusion_hyperparams["Alpha_bar"], \
                                self.diffusion_hyperparams["Sigma"] 


        diffusion_steps = t * torch.ones((x_t.shape[0], 1)).cuda()
        epsilon_theta = self.model((x_t, diffusion_steps))
        mu_theta = (x_t - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
        sigma_theta = Sigma[t]

        # sigma_theta = self.diffusion_hyperparams["Beta"][t].sqrt()
        
        return epsilon_theta, mu_theta, sigma_theta

    @torch.no_grad()
    def compute_eps_t(self, x_t: Union[torch.Tensor, np.ndarray], t):

        diffusion_steps = t * torch.ones((x_t.shape[0], 1)).cuda()
        epsilon_theta = self.model((x_t, diffusion_steps))

        return epsilon_theta

    def one_shot_denoise(self, x_t: Union[torch.Tensor, np.ndarray]):

        t = self.reverse_timestep - 1
        diffusion_steps = t * torch.ones((x_t.shape[0], 1)).cuda()
        epsilon_theta = self.model((x_t, diffusion_steps))

        pred_x_0 = self._predict_x0_from_eps(x_t, t, epsilon_theta)

        return pred_x_0
    
    def two_shot_denoise(self, x_t: Union[torch.Tensor, np.ndarray]):

        t = self.reverse_timestep - 1
        diffusion_steps = t * torch.ones((x_t.shape[0], 1)).cuda()
        epsilon_theta = self.model((x_t, diffusion_steps))

        pred_x_1 = self._predict_x1_from_eps(x_t, t, epsilon_theta)
        pred_x_0 = self._predict_x0_from_x1(pred_x_1)

        return pred_x_0

    def _predict_x0_from_eps(self, x_t, t, eps):

        assert x_t.shape == eps.shape

        Alpha_bar = self.diffusion_hyperparams["Alpha_bar"]

        sqrt_recip_alphas_bar = (1 / Alpha_bar).sqrt()
        sqrt_recipm1_alphas_bar = (1 / Alpha_bar - 1).sqrt()
        pred_x_0 = self._extract_into_tensor(sqrt_recip_alphas_bar, t, x_t.shape) * x_t - self._extract_into_tensor(sqrt_recipm1_alphas_bar, t, x_t.shape) * eps

        return pred_x_0

    def _predict_x1_from_eps(self, x_t, t, eps):

        Alpha = self.diffusion_hyperparams["Alpha"]
        Alpha_bar = self.diffusion_hyperparams["Alpha_bar"]
        Beta = self.diffusion_hyperparams["Beta"]

        mu = (Alpha_bar[t] / Alpha[0]).sqrt()
        sigma = (1 - Alpha_bar[t] - (Alpha_bar[t] / Alpha[0]) * Beta[0] ** 2).sqrt()

        pred_x_1 = (x_t - sigma * eps) / mu

        return pred_x_1
    
    def _predict_x0_from_x1(self, x_1):

        _, mu_0, _ = self.compute_coefficients(x_1, 0)

        pred_x_0 = mu_0

        return pred_x_0

    def _extract_into_tensor(self, arr_or_func, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.
        :param arr: the 1-D numpy array or a func.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        if callable(arr_or_func):
            res = arr_or_func(timesteps).float()
        else:
            if isinstance(arr_or_func, torch.Tensor):
                res = arr_or_func.cuda()[timesteps].float()
            elif isinstance(arr_or_func, np.ndarray):
                res = torch.from_numpy(arr_or_func).cuda()[timesteps].float()
            else:
                raise TypeError('Unsupported data type {} in arr_or_func'.format(type(arr_or_func)))
        
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
    
class ReffWave(torch.nn.Module):

    def __init__(self, 
                model: WaveNet_Speech_Commands, 
                diffusion_hyperparams: dict,
                reverse_timestep: int=200, 
                num_re: int=5
                ):
        super().__init__()

        '''
            model: input (x_t, t), output epsilon_theta at timestep t
        '''

        self.model = model
        self.diffusion_hyperparams = diffusion_hyperparams
        self.reverse_timestep = reverse_timestep
        self.freeze = False
        self.num_re = num_re

    def forward(self, waveforms: Union[torch.Tensor, np.ndarray]):

        if isinstance(waveforms, np.ndarray): 
            waveforms = torch.from_numpy(waveforms)

        output = waveforms
        # with torch.no_grad():
        for i in range(self.num_re):
            output = self.diffusion(output)
            output = self.one_shot_denoise(output)

        return output

    def diffusion(self, x_0: Union[torch.Tensor, np.ndarray]) -> torch.Tensor: 
        
        '''convert np.array to torch.tensor'''
        if isinstance(x_0, np.ndarray): 
            x_0 = torch.from_numpy(x_0)

        T, Alpha, Alpha_bar, Sigma = self.diffusion_hyperparams["T"], \
                                    self.diffusion_hyperparams["Alpha"], \
                                    self.diffusion_hyperparams["Alpha_bar"], \
                                    self.diffusion_hyperparams["Sigma"] 
        assert len(Alpha) == T
        assert len(Alpha_bar) == T
        assert len(Sigma) == T
        assert x_0.ndim == 3

        '''noising'''
        z = torch.normal(0, 1, size=x_0.shape).cuda()
        x_t = torch.sqrt(Alpha_bar[self.reverse_timestep-1]).cuda() * x_0 + torch.sqrt(1-Alpha_bar[self.reverse_timestep-1]).cuda() * z

        return x_t

    def one_shot_denoise(self, x_t: Union[torch.Tensor, np.ndarray]):

        t = self.reverse_timestep - 1
        diffusion_steps = t * torch.ones((x_t.shape[0], 1)).cuda()
        epsilon_theta = self.model((x_t, diffusion_steps))

        pred_x_0 = self._predict_x0_from_eps(x_t, t, epsilon_theta)

        return pred_x_0

    def _predict_x0_from_eps(self, x_t, t, eps):

        assert x_t.shape == eps.shape

        Alpha_bar = self.diffusion_hyperparams["Alpha_bar"]

        sqrt_recip_alphas_bar = (1 / Alpha_bar).sqrt()
        sqrt_recipm1_alphas_bar = (1 / Alpha_bar - 1).sqrt()
        pred_x_0 = self._extract_into_tensor(sqrt_recip_alphas_bar, t, x_t.shape) * x_t - self._extract_into_tensor(sqrt_recipm1_alphas_bar, t, x_t.shape) * eps

        return pred_x_0

    def _extract_into_tensor(self, arr_or_func, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.
        :param arr: the 1-D numpy array or a func.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        if callable(arr_or_func):
            res = arr_or_func(timesteps).float()
        else:
            if isinstance(arr_or_func, torch.Tensor):
                res = arr_or_func.cuda()[timesteps].float()
            elif isinstance(arr_or_func, np.ndarray):
                res = torch.from_numpy(arr_or_func).cuda()[timesteps].float()
            else:
                raise TypeError('Unsupported data type {} in arr_or_func'.format(type(arr_or_func)))
        
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
    # not used yet
    # region
    # def fast_denoise(self, waveforms: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        
    #     T = self.diffusion_hyperparams["T"]
    #     Eta = torch.tensor([0.0001, 0.001, 0.01, 0.05, 0.2, 0.7])

    #     self.T_fast = int((self.reverse_timestep / T) * len(Eta)) + 1
    #     # self.T_fast = 6
    #     self.Eta = Eta[self.T_fast:]
    #     self.Gamma = 1 - self.Eta
    #     self.Gamma_bar = torch.cumprod(self.Gamma, dim=0)

    #     x = waveforms.clone()

    #     for s in range(self.T_fast-1, -1, -1):
    #         t_s_align = self.compute_aligned_timestep(s).cuda()
    #         diffusion_steps = t_s_align * torch.ones((x.shape[0], 1)).cuda()
    #         epsilon_theta_fast = self.model((x, diffusion_steps))
    #         mu_theta_fast = (x - (self.Eta[s] / (1 - self.Gamma_bar[s]).sqrt()) * epsilon_theta_fast) / self.Gamma[s].sqrt()
            
    #         if s > 0:
    #             sigma_theta_fast = ((1 - self.Gamma_bar[s-1]) / (1 - self.Gamma_bar[s])) * self.Eta[s]
    #         else:
    #             sigma_theta_fast = self.Eta[0]
            
    #         x = mu_theta_fast + sigma_theta_fast * torch.normal(0, 1, size=x.shape).cuda()

    #     return x

    # def compute_aligned_timestep(self, s: int):

    #     T = self.diffusion_hyperparams["T"]
    #     Alpha_bar = self.diffusion_hyperparams["Alpha_bar"]
    #     gamma_bar_s = self.Gamma_bar[s]

    #     for t in range(T-1): 
    #         if Alpha_bar[t+1].sqrt() <= gamma_bar_s.sqrt() and gamma_bar_s.sqrt() <= Alpha_bar[t].sqrt():
    #             break
    #     assert Alpha_bar[t+1].sqrt() <= gamma_bar_s.sqrt() and gamma_bar_s.sqrt() <= Alpha_bar[t].sqrt()

    #     t_s_align = t + (Alpha_bar[t].sqrt() - gamma_bar_s.sqrt()) / (Alpha_bar[t].sqrt() - Alpha_bar[t+1].sqrt())

    #     return t_s_align
    # endregion

def create_diffwave_model(model_path, config_path, reverse_timestep=25):

    with open(config_path) as f:
        data = f.read()
    cfg = json.loads(data)

    wavenet_config = cfg["wavenet_config"]      # to define wavenet
    diffusion_config = cfg["diffusion_config"]    # basic hyperparameters
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)

    WaveNet_model = WaveNet_Speech_Commands(**wavenet_config).cuda()
    checkpoint = torch.load(model_path)
    WaveNet_model.load_state_dict(checkpoint['model_state_dict'])

    Denoiser = DiffWave(model=WaveNet_model, diffusion_hyperparams=diffusion_hyperparams, reverse_timestep=reverse_timestep)

    return Denoiser


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.json', 
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    parser.add_argument('--gpu', type=str, default='2')
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    # train_config            = config["train_config"]        # training parameters
    # global dist_config
    # dist_config             = config["dist_config"]         # to initialize distributed training
    global wavenet_config
    wavenet_config          = config["wavenet_config"]      # to define wavenet
    global diffusion_config
    diffusion_config        = config["diffusion_config"]    # basic hyperparameters
    global trainset_config
    trainset_config         = config["trainset_config"]     # to load trainset
    global diffusion_hyperparams 
    diffusion_hyperparams   = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters


    data_loader = load_Qualcomm_keyword(path=trainset_config['data_path'], batch_size=2)

    Net = DiffWave(**wavenet_config).cuda()
    model_path = 'DiffWave_Unconditional/exp/ch256_T200_betaT0.02/logs/checkpoint/1000000.pkl'
    checkpoint = torch.load(model_path)
    Net.load_state_dict(checkpoint['model_state_dict'])

    Denoiser = DiffWave(model=Net, diffusion_hyperparams=diffusion_hyperparams, reverse_timestep=25)

    

    
