import torch
from torch.nn.modules.loss import _Loss
from torch.nn import CrossEntropyLoss, CTCLoss

from typing import TYPE_CHECKING, Optional, Tuple, Union
import numpy as np
import scipy.signal as ss

def project_to_norm_ball(x: Union[torch.Tensor, np.ndarray], p: str, eps: float):

    if p == 'linf':
        x_ = torch.clamp(x, -eps, eps)
    elif p == 'l2':
        norm_x = torch.norm(input=x, dim=(1,2))[:,None,None]
        factor = torch.min(torch.ones_like(norm_x), eps / norm_x)
        x_ = x * factor
    else:
        raise NotImplementedError(f'Unsupported norm: {p}!')
    
    return x_

def lp_norm(x: Union[torch.Tensor, np.ndarray], p: str):

    if p == 'linf':
        norm = torch.max(torch.abs(x))
    elif p == 'l2':
        if x.ndim == 3:
            norm = torch.norm(input=x, dim=(1,2))[:,None,None]
        elif x.ndim == 2: 
            norm = torch.norm(input=x, dim=(1,))
    else:
        raise NotImplementedError(f'Unsupported norm: {p}!')
    
    return norm

class PsychoacousticMasker:
    """
    Implements psychoacoustic model of Lin and Abdulla (2015) following Qin et al. (2019) simplifications.

    | Paper link: Lin and Abdulla (2015), https://www.springer.com/gp/book/9783319079738
    | Paper link: Qin et al. (2019), http://proceedings.mlr.press/v97/qin19a.html
    """

    def __init__(self, window_size: int = 2048, hop_size: int = 512, sample_rate: int = 16000) -> None:
        """
        Initialization.

        :param window_size: Length of the window. The number of STFT rows is `(window_size // 2 + 1)`.
        :param hop_size: Number of audio samples between adjacent STFT columns.
        :param sample_rate: Sampling frequency of audio inputs.
        """
        self._window_size = window_size
        self._hop_size = hop_size
        self._sample_rate = sample_rate

        # init some private properties for lazy loading
        self._fft_frequencies: Optional[np.ndarray] = None
        self._bark: Optional[np.ndarray] = None
        self._absolute_threshold_hearing: Optional[np.ndarray] = None

    def calculate_threshold_and_psd_maximum(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the global masking threshold for an audio input and also return its maximum power spectral density.

        This method is the main method to call in order to obtain global masking thresholds for an audio input. It also
        returns the maximum power spectral density (PSD) for each frame. Given an audio input, the following steps are
        performed:

        1. STFT analysis and sound pressure level normalization
        2. Identification and filtering of maskers
        3. Calculation of individual masking thresholds
        4. Calculation of global masking thresholds

        :param audio: Audio samples of shape `(length,)`.
        :return: Global masking thresholds of shape `(window_size // 2 + 1, frame_length)` and the PSD maximum for each
            frame of shape `(frame_length)`.
        """
        psd_matrix, psd_max = self.power_spectral_density(audio)
        threshold = np.zeros_like(psd_matrix)
        for frame in range(psd_matrix.shape[1]):
            # apply methods for finding and filtering maskers
            maskers, masker_idx = self.filter_maskers(*self.find_maskers(psd_matrix[:, frame]))
            # apply methods for calculating global threshold
            threshold[:, frame] = self.calculate_global_threshold(
                self.calculate_individual_threshold(maskers, masker_idx)
            )
        return threshold, psd_max

    @property
    def window_size(self) -> int:
        """
        :return: Window size of the masker.
        """
        return self._window_size

    @property
    def hop_size(self) -> int:
        """
        :return: Hop size of the masker.
        """
        return self._hop_size

    @property
    def sample_rate(self) -> int:
        """
        :return: Sample rate of the masker.
        """
        return self._sample_rate

    @property
    def fft_frequencies(self) -> np.ndarray:
        """
        :return: Discrete fourier transform sample frequencies.
        """
        if self._fft_frequencies is None:
            self._fft_frequencies = np.linspace(0, self.sample_rate / 2, self.window_size // 2 + 1)
        return self._fft_frequencies

    @property
    def bark(self) -> np.ndarray:
        """
        :return: Bark scale for discrete fourier transform sample frequencies.
        """
        if self._bark is None:
            self._bark = 13 * np.arctan(0.00076 * self.fft_frequencies) + 3.5 * np.arctan(
                np.square(self.fft_frequencies / 7500.0)
            )
        return self._bark

    @property
    def absolute_threshold_hearing(self) -> np.ndarray:
        """
        :return: Absolute threshold of hearing (ATH) for discrete fourier transform sample frequencies.
        """
        if self._absolute_threshold_hearing is None:
            # ATH applies only to frequency range 20Hz<=f<=20kHz
            # note: deviates from Qin et al. implementation by using the Hz range as valid domain
            valid_domain = np.logical_and(20 <= self.fft_frequencies, self.fft_frequencies <= 2e4)
            freq = self.fft_frequencies[valid_domain] * 0.001

            # outside valid ATH domain, set values to -np.inf
            # note: This ensures that every possible masker in the bins <=20Hz is valid. As a consequence, the global
            # masking threshold formula will always return a value different to np.inf
            self._absolute_threshold_hearing = np.ones(valid_domain.shape) * -np.inf

            self._absolute_threshold_hearing[valid_domain] = (
                3.64 * pow(freq, -0.8) - 6.5 * np.exp(-0.6 * np.square(freq - 3.3)) + 0.001 * pow(freq, 4) - 12
            )
        return self._absolute_threshold_hearing

    def power_spectral_density(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the power spectral density matrix for an audio input.

        :param audio: Audio sample of shape `(length,)`.
        :return: PSD matrix of shape `(window_size // 2 + 1, frame_length)` and maximum vector of shape
        `(frame_length)`.
        """
        import librosa

        # compute short-time Fourier transform (STFT)
        audio_float = audio.astype(np.float32)
        stft_params = {
            "n_fft": self.window_size,
            "hop_length": self.hop_size,
            "win_length": self.window_size,
            "window": ss.get_window("hann", self.window_size, fftbins=True),
            "center": False,
        }
        stft_matrix = librosa.core.stft(audio_float, **stft_params)

        # compute power spectral density (PSD)
        with np.errstate(divide="ignore"):
            gain_factor = np.sqrt(8.0 / 3.0)
            psd_matrix = 20 * np.log10(np.abs(gain_factor * stft_matrix / self.window_size))
            psd_matrix = psd_matrix.clip(min=-200)

        # normalize PSD at 96dB
        psd_matrix_max = np.max(psd_matrix)
        psd_matrix_normalized = 96.0 - psd_matrix_max + psd_matrix

        return psd_matrix_normalized, psd_matrix_max

    @staticmethod
    def find_maskers(psd_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify maskers.

        Possible maskers are local PSD maxima. Following Qin et al., all maskers are treated as tonal. Thus neglecting
        the nontonal type.

        :param psd_vector: PSD vector of shape `(window_size // 2 + 1)`.
        :return: Possible PSD maskers and indices.
        """
        # identify maskers. For simplification it is assumed that all maskers are tonal (vs. nontonal).
        masker_idx = ss.argrelmax(psd_vector)[0]

        # smooth maskers with their direct neighbors
        psd_maskers = 10 * np.log10(np.sum([10 ** (psd_vector[masker_idx + i] / 10) for i in range(-1, 2)], axis=0))
        return psd_maskers, masker_idx

    def filter_maskers(self, maskers: np.ndarray, masker_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter maskers.

        First, discard all maskers that are below the absolute threshold of hearing. Second, reduce pairs of maskers
        that are within 0.5 bark distance of each other by keeping the larger masker.

        :param maskers: Masker PSD values.
        :param masker_idx: Masker indices.
        :return: Filtered PSD maskers and indices.
        """
        # filter on the absolute threshold of hearing
        # note: deviates from Qin et al. implementation by filtering first on ATH and only then on bark distance
        ath_condition = maskers > self.absolute_threshold_hearing[masker_idx]
        masker_idx = masker_idx[ath_condition]
        maskers = maskers[ath_condition]

        # filter on the bark distance
        bark_condition = np.ones(masker_idx.shape, dtype=bool)
        i_prev = 0
        for i in range(1, len(masker_idx)):
            # find pairs of maskers that are within 0.5 bark distance of each other
            if self.bark[i] - self.bark[i_prev] < 0.5:
                # discard the smaller masker
                i_todelete, i_prev = (i_prev, i_prev + 1) if maskers[i_prev] < maskers[i] else (i, i_prev)
                bark_condition[i_todelete] = False
            else:
                i_prev = i
        masker_idx = masker_idx[bark_condition]
        maskers = maskers[bark_condition]

        return maskers, masker_idx

    def calculate_individual_threshold(self, maskers: np.ndarray, masker_idx: np.ndarray) -> np.ndarray:
        """
        Calculate individual masking threshold with frequency denoted at bark scale.

        :param maskers: Masker PSD values.
        :param masker_idx: Masker indices.
        :return: Individual threshold vector of shape `(window_size // 2 + 1)`.
        """
        delta_shift = -6.025 - 0.275 * self.bark
        threshold = np.zeros(masker_idx.shape + self.bark.shape)
        # TODO reduce for loop
        for k, (masker_j, masker) in enumerate(zip(masker_idx, maskers)):
            # critical band rate of the masker
            z_j = self.bark[masker_j]
            # distance maskees to masker in bark
            delta_z = self.bark - z_j
            # define two-slope spread function:
            #   if delta_z <= 0, spread_function = 27*delta_z
            #   if delta_z > 0, spread_function = [-27+0.37*max(PSD_masker-40,0]*delta_z
            spread_function = 27 * delta_z
            spread_function[delta_z > 0] = (-27 + 0.37 * max(masker - 40, 0)) * delta_z[delta_z > 0]

            # calculate threshold
            threshold[k, :] = masker + delta_shift[masker_j] + spread_function
        return threshold

    def calculate_global_threshold(self, individual_threshold):
        """
        Calculate global masking threshold.

        :param individual_threshold: Individual masking threshold vector.
        :return: Global threshold vector of shape `(window_size // 2 + 1)`.
        """
        # note: deviates from Qin et al. implementation by taking the log of the summation, which they do for numerical
        #       stability of the stage 2 optimization. We stabilize the optimization in the loss itself.
        with np.errstate(divide="ignore"):
            return 10 * np.log10(
                np.sum(10 ** (individual_threshold / 10), axis=0) + 10 ** (self.absolute_threshold_hearing / 10)
            )

class AudioAttack():
    '''
        Qin-Imperceptible & CW-Audio WhiteBox attack;
        the first attack stage is PGD.
    '''
    def __init__(
        self,
        model: torch.nn.Module,
        masker: "PsychoacousticMasker" = None,
        criterion: "_Loss" = CrossEntropyLoss(),  
        eps: float = 2000.0,
        norm: str = 'linf', 
        learning_rate_1: float = 100.0,
        max_iter_1: int = 1000,
        alpha: float = 0.05,
        learning_rate_2: float = 1.0,
        max_iter_2: int = 4000,
        loss_theta_min: float = 0.05,
        decrease_factor_eps: float = 0.8,
        num_iter_decrease_eps: int = 10,
        increase_factor_alpha: float = 1.2,
        num_iter_increase_alpha: int = 20,
        decrease_factor_alpha: float = 0.8,
        num_iter_decrease_alpha: int = 50,
        eot_attack_size: int=15,
        eot_defense_size: int=15,
        verbose: int=1
        ) -> None:

        self.model = model
        self.masker = masker
        self.criterion = criterion

        self.eps = eps
        self.norm = norm

        self.learning_rate_1 = learning_rate_1
        self.max_iter_1 = max_iter_1
        self.alpha = alpha
        self.learning_rate_2 = learning_rate_2
        self.max_iter_2 = max_iter_2

        self._targeted = True
        self.loss_theta_min = loss_theta_min
        self.decrease_factor_eps = decrease_factor_eps
        self.num_iter_decrease_eps = num_iter_decrease_eps
        self.increase_factor_alpha = increase_factor_alpha
        self.num_iter_increase_alpha = num_iter_increase_alpha
        self.decrease_factor_alpha = decrease_factor_alpha
        self.num_iter_decrease_alpha = num_iter_decrease_alpha
        
        self.scale_factor = 2**-15

        self.eot_attack_size = eot_attack_size
        self.eot_defense_size = eot_defense_size

        self.verbose = verbose

        if self.eot_attack_size > 1 or self.eot_defense_size > 1:
            from ._EOT import EOT
            self.eot_model = EOT(model=model, loss=self.criterion, EOT_size=eot_attack_size)

        if self.masker is not None:
            self._window_size = self.masker.window_size
            self._hop_size = self.masker.hop_size
            self._sample_rate = self.masker.sample_rate

    
    def generate(self, x: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray], targeted: bool=True):
        
        self._targeted = targeted

        '''convert np.array to torch.tensor'''
        if isinstance(x, np.ndarray): 
            x = torch.from_numpy(x)
        if isinstance(y, np.ndarray): 
            y = torch.from_numpy(y)
        
        x_adv, success_stage_1 = self.stage_1(x, y)

        '''stage 2 not used'''
        if self.max_iter_2 > 0:
            x_adv, success_stage_2 = self.stage_2(x, x_adv, y)
            return x_adv, (success_stage_1, success_stage_2)
        else:
            return x_adv, (success_stage_1, None)

    def stage_1(self, x: torch.Tensor, y: torch.Tensor):
        
        '''
            x: waveform tensor
            y: target
        '''
        if x.dtype == torch.float32:
            eps = self.eps
            lr = self.learning_rate_1
        else: 
            # if 16-bit data
            eps = self.eps * 2**15
            lr = self.learning_rate_1 * 2**15

        batch_size = x.shape[0]
        x_adv = [None] * batch_size
        delta = torch.zeros_like(x, requires_grad=True)
        epsilon = [eps] * batch_size
        
        for i in range(0, self.max_iter_1 + 1):
            
            '''update perturbed inputs and predictions'''
            x_pert = x + delta

            if self.eot_defense_size > 1:
                self.eot_model.EOT_size = self.eot_defense_size
                # self.eot_model.EOT_batch_size = batch_size
                self.eot_model.use_grad = False
                y_pert, _, _, _ = self.eot_model(x_pert, y)
            else:
                y_pert = self.model(x_pert)

            ''' 
                save current best adv example. 
                if the prediction of a clean example is wrong, note it as an adv example.
                (for the convenience of computing robust acc)
            '''
            prediction = y_pert.max(1, keepdim=True)[1]
            if self._targeted: 
                for j in range(batch_size):
                    if prediction[j] == y[j]:
                        x_adv[j] = x_pert[j]
            else: 
                for j in range(batch_size):
                    if prediction[j] != y[j]:
                        x_adv[j] = x_pert[j]

            # '''decrease max norm bound epsilon if attack succeeds'''
            # if i % self.num_iter_decrease_eps == 0 and i > 0:
            #     if self._targeted: 
            #         for j in range(batch_size):
            #             if prediction[j] == y[j]:
            #                 perturbation_norm = lp_norm(delta.data[j], p=self.norm).item() #torch.max(torch.abs(delta.data[j]))
            #                 if epsilon[j] > perturbation_norm:
            #                     epsilon[j] = perturbation_norm
            #                 epsilon[j] *= self.decrease_factor_eps
            #     else:
            #         for j in range(batch_size):
            #             if prediction[j] != y[j]:
            #                 perturbation_norm = lp_norm(delta.data[j], p=self.norm).item() #torch.max(torch.abs(delta.data[j]))
            #                 if epsilon[j] > perturbation_norm:
            #                     epsilon[j] = perturbation_norm
            #                 epsilon[j] *= self.decrease_factor_eps
            
            if i == self.max_iter_1:
                break

            '''compute gradients'''

            if self.eot_attack_size > 1:
                self.eot_model.EOT_size = self.eot_attack_size
                # self.eot_model.EOT_batch_size = batch_size
                self.eot_model.use_grad = True
                _, _, grad, _ = self.eot_model(x_pert, y)
            else: 
                loss = self.criterion(y_pert, y)
                loss.backward()
                grad = delta.grad

            '''update perturbations'''
            if self._targeted:
                # delta.data = delta.data - lr * delta.grad.data.sign()
                delta.data = delta.data - lr * grad.data.sign()
            else:
                # delta.data = delta.data + lr * delta.grad.data.sign()
                if self.norm == 'linf':
                    delta.data = delta.data + lr * grad.data.sign()
                elif self.norm == 'l2':
                    delta.data = delta.data + lr * grad.data / lp_norm(grad.data, p=self.norm)
            # delta.data = torch.cat([torch.clamp(torch.unsqueeze(p, 1), -e, e) for p, e in zip(delta.data, epsilon)], dim=0)
            delta.data = torch.cat([project_to_norm_ball(torch.unsqueeze(p, 1), self.norm, e) for p, e in zip(delta.data, epsilon)], dim=0)
            delta.data = (x + delta.data).clamp(-1,1) - x
            delta.grad.zero_()
        
        x_pert = x + delta
        # success_stage_1 = batch_size
        success_stage_1 = [True] * batch_size

        ''' return perturbed x if no adversarial example found '''
        for j in range(batch_size):
            if x_adv[j] is None:
                if self.verbose:
                    print("Adversarial attack stage 1 for x_{} was not successful".format(j))
                x_adv[j] = x_pert[j]
                # success_stage_1 = success_stage_1 - 1 
                success_stage_1[j] = False

        x_adv = torch.unsqueeze(torch.cat(x_adv, dim=0), 1)

        return x_adv, success_stage_1
    
    '''not used'''
    def stage_2(self, x: torch.Tensor, x_adv: torch.Tensor, y: torch.Tensor=None):

        """
        Create imperceptible, adversarial example with small perturbation.

        This method implements the part of the paper by Qin et al. (2019) that is described as the second stage of the
        attack. The resulting adversarial audio samples are able to successfully deceive the ASR estimator and are
        imperceptible to the human ear.

        :param x: An array with the original inputs to be attacked. Shape: (batch_size, n, length).
        :param x_adversarial: An array with the adversarial examples. Shape: (batch_size, n, length).
        :param y: Target values of shape (batch_size,). Each sample in `y` is a string and it may possess different
            lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :return: An array with the imperceptible, adversarial outputs.
        """

        if x.dtype == torch.float32:
            lr = self.scale_factor * self.learning_rate_2
        else: 
            lr = self.learning_rate_2

        batch_size = x.shape[0]
        alpha_min = 0.0005

        early_stop = [False] * batch_size
        alpha = torch.tensor([self.alpha] * batch_size, dtype=torch.float32).to(x.device)
        loss_theta_previous = [torch.inf] * batch_size
        loss_theta = [torch.inf] * batch_size
        x_imperceptible = [None] * batch_size

        # if inputs are *not* ragged, we can't multiply alpha * gradients_theta
        if x.ndim != 1:
            # alpha = torch.unsqueeze(alpha, axis=-1)
            alpha = alpha[:,None,None]

        masking_threshold, psd_maximum = self._stabilized_threshold_and_psd_maximum(x)

        delta = torch.zeros_like(x, requires_grad=True)
        delta.data = x_adv.data - x.data

        for i in range(0, self.max_iter_2 + 1):
            
            '''update perturbed inputs and predictions'''
            x_pert = x + delta
            y_adv = self.model(x_pert)

            ''' 
                save current best imp adv example. 
                (for the convenience of computing attack success rate)
            '''
            prediction = y_adv.max(1, keepdim=True)[1]
            if self._targeted: 
                for j in range(batch_size):
                    if prediction[j] == y[j] and loss_theta[j] < loss_theta_previous[j]:
                        x_imperceptible[j] = x_pert[j]
                        loss_theta_previous[j] = loss_theta[j]
            else: 
                for j in range(batch_size):
                    if prediction[j] != y[j] and loss_theta[j] < loss_theta_previous[j]:
                        x_imperceptible[j] = x_pert[j]
                        loss_theta_previous[j] = loss_theta[j]

            '''update alpha'''
            if (i % self.num_iter_increase_alpha == 0 or i % self.num_iter_decrease_alpha == 0) and i > 0:

                if self._targeted:
                    for j in range(batch_size):
                        # validate if adversarial target succeeds, i.e. f(x_perturbed)==y
                        if i % self.num_iter_increase_alpha == 0 and prediction[j] == y[j]:
                            # increase alpha
                            alpha[j] *= self.increase_factor_alpha
                        # validate if adversarial target fails, i.e. f(x_perturbed)!=y
                        if i % self.num_iter_decrease_alpha == 0 and prediction[j] != y[j]:
                            # decrease alpha
                            alpha[j] = max(alpha[j] * self.decrease_factor_alpha, alpha_min)
                    
                else:
                    for j in range(batch_size):
                        # validate if adversarial target succeeds, i.e. f(x_perturbed)!=y
                        if i % self.num_iter_increase_alpha == 0 and prediction[j] != y[j]:
                            # increase alpha
                            alpha[j] *= self.increase_factor_alpha
                        # validate if adversarial target fails, i.e. f(x_perturbed)==y
                        if i % self.num_iter_decrease_alpha == 0 and prediction[j] == y[j]:
                            # decrease alpha
                            alpha[j] = max(alpha[j] * self.decrease_factor_alpha, alpha_min)

            if i == self.max_iter_2:
                break
            
            '''compute gradients'''
            loss = self.criterion(y_adv, y)
            loss.backward()
            gradients_net = delta.grad.data
            gradients_theta, loss_theta = self._loss_gradient_masking_threshold(
                delta, x, masking_threshold, psd_maximum
            )
            '''update perturbations'''
            assert gradients_net.shape == gradients_theta.shape
            if self._targeted:
                delta.data = delta.data - lr * (gradients_net + alpha * gradients_theta)
            else:
                delta.data = delta.data + lr * (gradients_net + alpha * gradients_theta)
            delta.data = (x + delta.data).clamp(-1,1) - x
            
            # note: avoids nan values in loss theta, which can occur when loss converges to zero.
            for j in range(batch_size):
                if loss_theta[j] < self.loss_theta_min and not early_stop[j]:
                    if self.verbose:
                        print(
                            "Batch sample {} reached minimum threshold of {} for theta loss.".format(j, self.loss_theta_min)
                            )
                    early_stop[j] = True
            if all(early_stop):
                if self.verbose:
                    print(
                        "All batch samples reached minimum threshold for theta loss. Stopping early at iteration {}".format(i)
                        )
                break

        # return perturbed x if no adversarial example found
        # success_stage_2 = batch_size
        success_stage_2 = [True] * batch_size

        for j in range(batch_size):
            if x_imperceptible[j] is None:
                if self.verbose:
                    print("Adversarial attack stage 2 for x_{} was not successful".format(j))
                x_imperceptible[j] = x_pert[j]
                # success_stage_2 = success_stage_2 - 1 
                success_stage_2[j] = False

        x_imperceptible = torch.unsqueeze(torch.cat(x_imperceptible, dim=0), 1)

        return x_imperceptible, success_stage_2
    
    def _loss_gradient_masking_threshold(
            self,
            perturbation: torch.Tensor,
            x: torch.Tensor,
            masking_threshold_stabilized: torch.Tensor,
            psd_maximum_stabilized: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute loss gradient of the global masking threshold w.r.t. the PSD approximate of the perturbation.

        The loss is defined as the hinge loss w.r.t. to the frequency masking threshold of the original audio input `x`
        and the normalized power spectral density estimate of the perturbation. In order to stabilize the optimization
        problem during back-propagation, the `10*log`-terms are canceled out.

        :param perturbation: Adversarial perturbation.
        :param x: An array with the original inputs to be attacked.
        :param masking_threshold_stabilized: Stabilized masking threshold for the original input `x`.
        :param psd_maximum_stabilized: Stabilized maximum across frames, i.e. shape is `(batch_size, frame_length)`, of
            the original unnormalized PSD of `x`.
        :return: Tuple consisting of the loss gradient, which has same shape as `perturbation`, and loss value.
        """
        # pad input
        perturbation_padded, delta_mask = self.pad_sequence_input(perturbation.detach().cpu().numpy()) # shape: (batch_size, length)
        perturbation_padded, delta_mask = torch.from_numpy(perturbation_padded).to(x.device), torch.from_numpy(delta_mask).to(x.device)
        
        perturbation_padded.requires_grad = True
        psd_perturbation = self._approximate_power_spectral_density(perturbation_padded, psd_maximum_stabilized)
        loss = torch.mean(torch.nn.ReLU()(psd_perturbation - masking_threshold_stabilized), dim=(1, 2), keepdims=False)
        loss.sum().backward()

        gradients_padded = perturbation_padded.grad.detach()
        loss_value = loss.detach()

        # undo padding, i.e. change gradients shape from (nb_samples, max_length) to (nb_samples)
        lengths = delta_mask.sum(axis=1)
        gradients = []
        for gradient_padded, length in zip(gradients_padded, lengths):
            gradient = gradient_padded[None, :length] # shape: (1, length)
            gradients.append(gradient)

        gradients = torch.unsqueeze(torch.cat(gradients, dim=0), 1) # shape: (batch_size, 1, length)
        return gradients, loss_value

    def _approximate_power_spectral_density(
        self, perturbation: torch.Tensor, psd_maximum_stabilized: torch.Tensor
        ) -> torch.Tensor:
        """
        Approximate the power spectral density for a perturbation `perturbation` in PyTorch.

        See also `ImperceptibleASR._approximate_power_spectral_density_tf`.
        """
        # compute short-time Fourier transform (STFT)
        # pylint: disable=W0212
        stft_matrix = torch.stft(
            perturbation,
            n_fft=self._window_size,
            hop_length=self._hop_size,
            win_length=self._window_size,
            center=False,
            window=torch.hann_window(self._window_size).to(perturbation.device),
        ).to(perturbation.device)

        # compute power spectral density (PSD)
        # note: fixes implementation of Qin et al. by also considering the square root of gain_factor
        gain_factor = np.sqrt(8.0 / 3.0)
        stft_matrix_abs = torch.sqrt(torch.sum(torch.square(gain_factor * stft_matrix / self._window_size), -1))
        psd_matrix = torch.square(stft_matrix_abs)

        # approximate normalized psd: psd_matrix_approximated = 10^((96.0 - psd_matrix_max + psd_matrix)/10)
        psd_matrix_approximated = pow(10.0, 9.6) / psd_maximum_stabilized.reshape(-1, 1, 1) * psd_matrix

        # return PSD matrix such that shape is (batch_size, window_size // 2 + 1, frame_length)
        return psd_matrix_approximated

    def _stabilized_threshold_and_psd_maximum(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return batch of stabilized masking thresholds and PSD maxima.

        :param x: An array with the original inputs to be attacked.
        :return: Tuple consisting of stabilized masking thresholds and PSD maxima.
        """
        masking_threshold = []
        psd_maximum = []
        x_padded, _ = self.pad_sequence_input(x.detach().cpu().numpy()) # shape: (batch_size, length, )

        for x_i in x_padded:
            m_t, p_m = self.masker.calculate_threshold_and_psd_maximum(x_i)
            masking_threshold.append(m_t)
            psd_maximum.append(p_m)
        # stabilize imperceptible loss by canceling out the "10*log" term in power spectral density maximum and
        # masking threshold
        masking_threshold_stabilized = 10 ** (np.array(masking_threshold) * 0.1) 
        psd_maximum_stabilized = 10 ** (np.array(psd_maximum) * 0.1)
        
        masking_threshold_stabilized = torch.from_numpy(masking_threshold_stabilized).to(x.device)
        psd_maximum_stabilized = torch.from_numpy(psd_maximum_stabilized).to(x.device)

        # masking_threshold_stabilized = torch.unsqueeze(masking_threshold_stabilized, dim=1)
        # psd_maximum_stabilized = torch.unsqueeze(psd_maximum_stabilized, dim=1)

        return masking_threshold_stabilized, psd_maximum_stabilized
    
    def pad_sequence_input(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply padding to a batch of 1-dimensional samples such that it has shape of (batch_size, max_length).

        :param x: A batch of 1-dimensional input data, e.g. `np.array([np.array([1,2,3]), np.array([4,5,6,7])])`.
        :return: The padded input batch and its corresponding mask.
        """
        if x.ndim > 2:
            x = np.squeeze(x, axis=1)

        max_length = max(map(len, x))
        batch_size = x.shape[0]

        # note: use dtype of inner elements
        x_padded = np.zeros((batch_size, max_length), dtype=x[0].dtype)
        x_mask = np.zeros((batch_size, max_length), dtype=bool)

        for i, x_i in enumerate(x):
            x_padded[i, : len(x_i)] = x_i
            x_mask[i, : len(x_i)] = 1
        return x_padded, x_mask


# '''Black-box Attack'''

# class SquareAttack():

#     def __init__(
#         self,
#         model: torch.nn.Module, 
#         eps, 
#         norm, 
#         n_iter, 
#         p_init
#         ) -> None:
        
#         self.model = model
#         self.eps = eps
#         self.norm = norm
#         self.n_iter = n_iter
#         self.p_init = p_init
#         # self._targeted = False
    
#     def loss(self, y: torch.Tensor, logits: torch.Tensor, targeted: bool=False, loss_type: str='margin'):

#         if loss_type == 'margin':
#             preds_correct_class = (logits * y).sum(1, keepdims=True)
#             diff = preds_correct_class - logits  # difference between the correct class and all other classes
#             diff[y] = torch.inf  # to exclude zeros coming from f_correct - f_correct
#             margin = diff.min(1, keepdims=True)
#             loss = margin * -1 if targeted else margin
#         elif loss_type == 'cross_entropy':
#             probs = torch.nn.Softmax(dim=1)(logits)
#             loss = -torch.log(probs[y])
#             loss = loss * -1 if not targeted else loss
#         else:
#             raise NotImplementedError(f'Unsupported loss type: {loss_type}!')

#         return loss.flatten()

#     def p_selection(self, p_init, it, n_iters):
#         """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
#         it = int(it / n_iters * 10000)

#         if 10 < it <= 50:
#             p = p_init / 2
#         elif 50 < it <= 200:
#             p = p_init / 4
#         elif 200 < it <= 500:
#             p = p_init / 8
#         elif 500 < it <= 1000:
#             p = p_init / 16
#         elif 1000 < it <= 2000:
#             p = p_init / 32
#         elif 2000 < it <= 4000:
#             p = p_init / 64
#         elif 4000 < it <= 6000:
#             p = p_init / 128
#         elif 6000 < it <= 8000:
#             p = p_init / 256
#         elif 8000 < it <= 10000:
#             p = p_init / 512
#         else:
#             p = p_init

#         return 
    
#     def generate(self, x: torch.Tensor, y: torch.Tensor, targeted: bool=False, p_init: float=0.1):

#         assert x.ndim == 3

#         min_val, max_val = -1, 1
#         n_features = x.shape[1] * x.shape[2]
#         n_ex_total = x.shape[0]
#         # x, y = x[corr_classified], y[corr_classified]

#         # init_delta = np.random.choice([-self.eps, self.eps], size=x.shape)
#         init_delta = (2 * torch.randint_like(x,1) - 1) * self.eps
#         x_best = (x + init_delta).clamp(min_val, max_val)
#         logits = self.model(x_best)

#         if not targeted:
#             loss_type = 'margin'
#         else:
#             loss_type = 'cross_entropy'
#         loss_min = self.loss(y, logits, targeted, loss_type).detach().cpu().numpy()
#         margin_min = self.loss(y, logits, targeted, 'margin').detach().cpu().numpy()
#         n_queries = np.ones(x.shape[0])  # ones because we have already used 1 query

#         time_start = time.time()
#         metrics = np.zeros([self.n_iter, 7])

#         for i_iter in range(self.n_iter - 1):
#             idx_to_fool = margin_min > 0
#             x_curr, x_best_curr, y_curr = x[idx_to_fool], x_best[idx_to_fool], y[idx_to_fool]
#             loss_min_curr, margin_min_curr = loss_min[idx_to_fool], margin_min[idx_to_fool]
#             delta = x_best_curr - x_curr

#             p = self.p_selection(p_init, i_iter, self.n_iter)

#             for i in range(x_best_curr.shape[0]):
#                 s = int(round(np.sqrt(p * n_features)))
#                 s = min(max(s, 1), x.shape[2]-1)  # at least 1 x 1 window is taken and at most 1 x length-1
#                 center_l = np.random.randint(0, x.shape[2] - s)

#                 x_curr_window = x_curr[i, :, center_l:center_l+s]
#                 x_best_curr_window = x_best_curr[i, :, center_l:center_l+s]

#                 # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
#                 while torch.sum(torch.abs((x_curr_window + delta[i, :, center_l:center_l+s]).clamp(min_val, max_val) - x_best_curr_window) < 10**-7) == s:
#                     delta[i, :, center_l:center_l+s] = (2 * torch.randint(0,1,[1,1]) - 1) * self.eps
            
#             x_new = (x_curr + delta).clamp(min_val, max_val)
#             logits = self.model(x_new)
#             loss = self.loss(y, logits, targeted, loss_type).detach().cpu().numpy()
#             margin = self.loss(y, logits, targeted, 'margin').detach().cpu().numpy()

#             idx_improved = loss < loss_min_curr
#             loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
#             margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr

#             idx_improved =idx_improved.reshape([-1, *[1]*len(x.shape[:-1])])
#             x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
#             n_queries[idx_to_fool] += 1

#             acc = (margin_min > 0.0).sum() / n_ex_total
#             acc_corr = (margin_min > 0.0).mean()
#             mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min <= 0]), np.median(n_queries[margin_min <= 0])
#             avg_margin_min = np.mean(margin_min)
#             time_total = time.time() - time_start
#             print('{}: acc={:.2%} acc_corr={:.2%} avg#q_ae={:.2f} med#q={:.1f}, avg_margin={:.2f} (n_ex={}, eps={:.3f}, {:.2f}s)'\
#                 .format(i_iter+1, acc, acc_corr, mean_nq_ae, median_nq_ae, avg_margin_min, x.shape[0], self.eps, time_total))
            
#             metrics[i_iter] = [acc, acc_corr, mean_nq, mean_nq_ae, median_nq_ae, margin_min.mean(), time_total]
#             # if (i_iter <= 500 and i_iter % 20 == 0) or (i_iter > 100 and i_iter % 50 == 0) or i_iter + 1 == self.n_iter or acc == 0:
#             #     np.save(metrics_path, metrics)
#             # if acc == 0:
#             #     break

#         return n_queries, x_best

# from utils import MarginalLoss
# class LinfSPSA():

#     '''
#         SPSA Attack (Uesato et al. 2018).
#         Based on: https://arxiv.org/abs/1802.05666
#         :param predict: predict function (single argument: input).
#         :param eps: the L_inf budget of the attack.
#         :param delta: scaling parameter of SPSA.
#         :param lr: the learning rate of the `Adam` optimizer.
#         :param nb_iter: number of iterations of the attack.
#         :param nb_sample: number of samples for SPSA gradient approximation.
#         :param max_batch_size: maximum batch size to be evaluated at once.
#         :param targeted: [description]
#         :param loss_fn: loss function (dual arguments: output, target).
#         :param clip_min: upper bound of image values.
#         :param clip_max: lower bound of image values.
#     '''

#     def __init__(self,
#                 model: torch.nn.Module,
#                 transform = None, 
#                 defender = None, 
#                 criterion: "_Loss" = MarginalLoss(),  
#                 eps: float = 2000.0,
#                 learning_rate: float = 100.0,
#                 max_iter: int = 1000,
#                 delta: float = 0.01,
#                 ) -> None:
        
#         self.model = model
#         self.transform = transform
#         self.defender = defender
#         self.criterion = criterion

#         self.eps = eps
#         self.learning_rate = learning_rate
#         self.max_iter = max_iter
#         self.delta = delta

#         self._targeted = True

#         self.scale_factor = 2**-15
    
     
#     def generate(self, 
#                 x: Union[torch.Tensor, np.ndarray], 
#                 y: Union[torch.Tensor, np.ndarray], 
#                 targeted: bool=True, 
#                 num_samples: int=512):
        
#         self._targeted = targeted

#         '''convert np.array to torch.tensor'''
#         if isinstance(x, np.ndarray): 
#             x = torch.from_numpy(x)
#         if isinstance(y, np.ndarray): 
#             y = torch.from_numpy(y)
        
#         x_adv = self.spsa_perturb(x, y, num_samples)

#         return x_adv

#     def spsa_perturb(self, x: torch.Tensor, y: torch.Tensor, num_samples: int=64):

#         dx = torch.zeros_like(x)
#         dx.grad = torch.zeros_like(dx)

#         if x.dtype == torch.float32:
#             eps = self.scale_factor * self.eps
#             lr = self.scale_factor * self.learning_rate
#         else: 
#             eps = self.eps
#             lr = self.learning_rate

#         optimizer = torch.optim.Adam([dx], lr=lr)

#         for i in range(self.max_iter):

#             optimizer.zero_grad()
#             if self._targeted:
#                 dx.grad = self.spsa_grad(x, y, num_samples)
#             else: 
#                 dx.grad = -self.spsa_grad(x, y, num_samples)
#             optimizer.step()
#             dx = dx.clamp(-eps, eps)

#         x_adv = x + dx
#         return x_adv

#     @torch.no_grad()
#     def predict(self, x: Union[torch.Tensor, np.ndarray]):

#         '''convert np.array to torch.tensor'''
#         if isinstance(x, np.ndarray): 
#             x = torch.from_numpy(x)

#         if self.defender is not None:
#             x = self.defender(x, grad_enable=False)
#         if self.transform is not None:
#             x = self.transform(x)
#         y = self.model(x)

#         return y
    
#     @torch.no_grad()
#     def spsa_grad(self, x: torch.Tensor, y: torch.Tensor, num_samples: int=512, batch_size: int=16):      
        
#         batches = [batch_size for _ in range(num_samples // batch_size)]
#         if num_samples % batch_size:
#             batches.append(num_samples % batch_size)

#         grad = torch.zeros_like(x)

#         for batch in batches:

#             x_batch = x.repeat(batch, 1, 1)
#             y_batch = y.repeat(batch)
#             v = torch.empty_like(x_batch)
#             v = v.bernoulli_().mul_(2.0).sub_(1.0)


#             df_n = self.criterion(self.predict(x_batch + self.delta * v), y_batch) \
#                 - self.criterion(self.predict(x_batch - self.delta * v), y_batch)
#             grad_n = df_n / (2. * self.delta * v)

#             grad = grad + sum(grad_n.split(x.shape[0], dim=0))

#         grad = grad / (num_samples * x.shape[0])

#         return grad