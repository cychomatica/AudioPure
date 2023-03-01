import torch
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import norm
import math

class RobustCertificate():

    def __init__(self, classifier: torch.nn.Module, transform=None, denoiser=None, one_shot_rev: bool=False, num_classes=10) -> None:

        self.classifier = classifier
        self.transform = transform
        self.denoiser = denoiser
        self.num_classes = num_classes
        self.one_shot_rev = one_shot_rev
        # self.model = torch.nn.Sequential(self.denoiser, self.transform, self.classifier)
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor):

        x_in = x
        
        if self.denoiser is not None:
            # x_in = self.denoiser.denoise(x_in)
            x_in = self.denoiser.one_shot_denoise(x_in)
            # x_in = self.denoiser.two_shot_denoise(x_in)

        if self.transform is not None:
            x_in = self.transform(x_in)

        output = self.classifier(x_in)
        return output

    @torch.no_grad()
    def smooth_predict(self, x: torch.Tensor, num_sampling: int=100, sigma=0.25, batch_size=64):

        assert(x.shape[0] == 1)

        batches = [batch_size for _ in range(num_sampling // batch_size)]
        if num_sampling % batch_size:
            batches.append(num_sampling % batch_size)

        output = []

        for batch in batches:

            x_in = x.repeat(batch, 1, 1)
            delta = torch.normal(0,sigma,size=x_in.shape).cuda()
            x_in = x_in + delta

            if self.denoiser is not None:
                alpha_bar_star = 1 / (1 + sigma**2)
                t_star = self.compute_t_star(alpha_bar_star)
                self.denoiser.reverse_timestep = t_star
                x_in = alpha_bar_star**0.5 * x_in

            output.append(self.forward(x_in))
        

        output = torch.cat(output, dim=0)
        num_classes = output.shape[-1]
        predictions = output.max(1, keepdim=True)[1].squeeze()

        counts = torch.zeros(num_classes, dtype=int)
        for i in range(num_classes):
            counts[i] = ((predictions==i).sum().item())

        return counts

    @torch.no_grad()
    def certify(self, 
                x: torch.Tensor, 
                y: torch.Tensor, 
                sigma: float=0.25, 
                n_0: int=100, 
                n: int=100000, 
                alpha: float=0.001,
                batch_size: int=64): 

        y_pred, radius = -torch.ones_like(y), torch.zeros_like(y,dtype=torch.float32)

        for i in range(x.shape[0]):

            x_in = x[i]
            counts_0 = self.smooth_predict(x_in, num_sampling=n_0, sigma=sigma, batch_size=batch_size)
            c_A = counts_0.max(0, keepdim=True)[1].item()

            counts = self.smooth_predict(x_in, num_sampling=n, sigma=sigma, batch_size=batch_size)

            pa = self.lower_conf_bound(k=counts[c_A], n=n, alpha=alpha)

            if pa > 0.5:
                y_pred[i] = c_A
                radius[i] = sigma * norm.ppf(pa)
            else:
                y_pred[i] = -1
                radius[i] = 0

        # correct = self.certified_robust_correct(y_pred, y, radius, r)

        return y_pred, radius, #correct

    def compute_t_star(self, alpha_bar_star):
        
        T = self.denoiser.diffusion_hyperparams['T']
        Alpha_bar = self.denoiser.diffusion_hyperparams['Alpha_bar']

        t_star = torch.abs(Alpha_bar- alpha_bar_star).min(0, keepdim=True)[1].item() + 1
        # t_star = (Alpha_bar[0] - alpha_star) / (Alpha_bar[0] - Alpha_bar[-1]) * (T - 1) + 1 # in [1, T]

        return t_star


    def lower_conf_bound(self, k, n, alpha=0.001):

        p = proportion_confint(k, n, alpha=2*alpha, method='beta')[0]

        return p

    def certified_robust_correct(self, y_pred: torch.Tensor, y_target: torch.Tensor, r_c: torch.Tensor, r: float=1.):

        correct = 0

        for i in range(len(y_pred)):

            if y_pred[i] == y_target[i] and r_c[i] >= r:
                correct += 1

        return correct