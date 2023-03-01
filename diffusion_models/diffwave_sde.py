import numpy as np
import torch
import torchsde

from .diffwave_ddpm import DiffWave, create_diffwave_model

def _extract_into_tensor(arr_or_func, timesteps, broadcast_shape):
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
        res = arr_or_func.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def restore_checkpoint(ckpt_dir, state, device):
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']


class RevVPSDE(torch.nn.Module):
    def __init__(self, model: DiffWave, score_type='ddpm', beta_min=0.02, beta_max=4, N=200,
                 audio_shape=(1, 16000), model_kwargs=None):
        """Construct a Variance Preserving SDE.

        Args:
          model: DiffWave model
          score_type: [guided_diffusion, score_sde, ddpm]
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: total steps
        """

        super().__init__()
        self.model = model
        self.score_type = score_type
        self.model_kwargs = model_kwargs
        self.audio_shape = audio_shape

        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min/N, beta_max/N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # self.alphas_cumprod_cont = lambda t: torch.exp(-0.5 * (self.beta_1 - self.beta_0) * t**2 - self.beta_0 * t)
        self.alphas_cumprod_cont = lambda t: torch.exp(-0.5 * self.N / (self.N - 1) * (self.beta_1 - self.beta_0) * t**2 - (self.N * self.beta_0 - self.beta_1) / (self.N - 1) * t)
        self.sqrt_1m_alphas_cumprod_neg_recip_cont = lambda t: -1. / torch.sqrt(1. - self.alphas_cumprod_cont(t))

        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def _scale_timesteps(self, t):
        assert torch.all(t <= 1) and torch.all(t >= 0), f't has to be in [0, 1], but get {t} with shape {t.shape}'
        return (t.float() * self.N).long()

    def vpsde_fn(self, t, x):
        # beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        # beta_t = self.beta_0 + (t * self.N - 1) / (self.N - 1) * (self.beta_1 - self.beta_0) # according to Song et al.
        disc_steps = self._scale_timesteps(t) - 1 # debug: using discrete params
        beta_t = self.discrete_betas[disc_steps].to(x.device) * self.N # debug: using discrete params

        drift = -0.5 * beta_t[:, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def rvpsde_fn(self, t, x, return_type='drift'):
        """Create the drift and diffusion functions for the reverse SDE"""
        drift, diffusion = self.vpsde_fn(t, x)

        if return_type == 'drift':

            assert x.ndim == 2 and np.prod(self.audio_shape) == x.shape[1], x.shape
            x_audio = x.view(-1, *self.audio_shape)

            if self.score_type == 'guided_diffusion':
                # model output is epsilon
                disc_steps = self._scale_timesteps(t) - 1
                epsilon_theta = self.model.compute_eps_t(x_audio, disc_steps[0])#t[0] * self.N - 1)  # x_audio.shape
                assert x_audio.shape == epsilon_theta.shape, f'{x_audio.shape}, {epsilon_theta.shape}'
                epsilon_theta = epsilon_theta.view(x.shape[0], -1)  # x_audio.shape -> (batch_size, )
                # score = _extract_into_tensor(self.sqrt_1m_alphas_cumprod_neg_recip_cont, t, x.shape) * epsilon_theta
                score = - epsilon_theta / self.sqrt_1m_alphas_cumprod[disc_steps[0]] # debug: using discrete params

            else:
                raise NotImplementedError(f'Unknown score type in RevVPSDE: {self.score_type}!')

            drift = drift - diffusion[:, None] ** 2 * score
            return drift

        else:
            disc_steps = self._scale_timesteps(t) - 1
            if disc_steps.unique() > 0:
                scale_factor = torch.sqrt(1 - self.alphas_cumprod[disc_steps - 1]) / torch.sqrt(1 - self.alphas_cumprod[disc_steps])
                scale_factor = scale_factor.to(x.device)
            else:
                scale_factor = 0
            # scale_factor = torch.sqrt(1. - self.alphas_cumprod_cont(t-1/self.N)) / torch.sqrt(1. - self.alphas_cumprod_cont(t))
            diffusion = scale_factor * diffusion
            return diffusion

    def f(self, t, x):
        """Create the drift function -f(x, 1-t) (by t' = 1 - t)
            sdeint only support a 2D tensor (batch_size, c*h*w)
        """
        t = t.expand(x.shape[0])  # (batch_size, )
        drift = self.rvpsde_fn(1 - t, x, return_type='drift')
        assert drift.shape == x.shape
        return -drift

    def g(self, t, x):
        """Create the diffusion function g(1-t) (by t' = 1 - t)
            sdeint only support a 2D tensor (batch_size, c*h*w)
        """
        t = t.expand(x.shape[0])  # (batch_size, )
        diffusion = self.rvpsde_fn(1 - t, x, return_type='diffusion')
        assert diffusion.shape == (x.shape[0], )
        return diffusion[:, None].expand(x.shape)

#explicitly adapted for DiffWave
class RevDiffWave(torch.nn.Module):
    def __init__(self, args, device=None):
        super().__init__()

        self.args = args
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        audio_shape = (1, 16000)

        #load ddpm
        model_path = args.ddpm_path
        model_config = args.ddpm_config
        print(f'model_config: {model_config}')
        model = create_diffwave_model(model_path=model_path, config_path=model_config, reverse_timestep=args.t)
        model.eval().to(self.device)
        self.T = 200 #total steps of ddpm

        self.model = model
        self.rev_vpsde = RevVPSDE(model=model, score_type=args.score_type, 
                                  beta_min=0.0001*self.T, beta_max=0.02*self.T,
                                  N=self.T, audio_shape=audio_shape,
                                  model_kwargs=None).to(self.device)
        self.betas = self.rev_vpsde.discrete_betas.float().to(self.device)


        print(f't: {args.t}, rand_t: {args.rand_t}, t_delta: {args.t_delta}')
        print(f'use_bm: {args.use_bm}')

    def audio_editing_sample(self, audio):

        '''
            audio should be in [-1, 1]
        '''

        assert isinstance(audio, torch.Tensor)
        batch_size = audio.shape[0]
        state_size = int(np.prod(audio.shape[1:]))


        assert audio.ndim == 3, audio.ndim
        audio = audio.to(self.device)
        x0 = audio

        xs = []
        for it in range(self.args.sample_step):

            e = torch.randn_like(x0).to(self.device)
            total_noise_levels = self.args.t
            if self.args.rand_t:
                total_noise_levels = self.args.t + np.random.randint(-self.args.t_delta, self.args.t_delta)
                print(f'total_noise_levels: {total_noise_levels}')
            a = (1 - self.betas).cumprod(dim=0).to(self.device)
            x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()


            epsilon_dt0, epsilon_dt1 = -1e-5, 1e-5
            t0, t1 = 1 - self.args.t / self.T + epsilon_dt0 , 1 - epsilon_dt1
            t_size = 2
            ts = torch.linspace(t0, t1, t_size).to(self.device)

            x_ = x.view(batch_size, -1)  # (batch_size, state_size)
            if self.args.use_bm:
                bm = torchsde.BrownianInterval(t0=t0, t1=t1, size=(batch_size, state_size), device=self.device)
                xs_ = torchsde.sdeint_adjoint(self.rev_vpsde, x_, ts, method='euler', bm=bm, dt=1./self.T)
            else:
                xs_ = torchsde.sdeint_adjoint(self.rev_vpsde, x_, ts, method='euler', dt=1./self.T)
            x0 = xs_[-1].view(x.shape)  # (batch_size, 1, 16000)

            # x_ddpm_0 = self.model.reverse(x)
            # x0 = self.model._predict_x0_from_x1(x0) # try

            xs.append(x0)

        return torch.cat(xs, dim=0)

    def forward(self, x):

        output = self.audio_editing_sample(x)

        return output
        
