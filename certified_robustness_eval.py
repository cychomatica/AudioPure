import os
import argparse

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import *
import torchaudio

import numpy as np
import matplotlib.pyplot as plt

import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    '''SC09 classifier arguments'''
    parser.add_argument("--data_path", help='sc09 dataset folder')
    parser.add_argument("--classifier_path", help='dir of saved classifier model')
    parser.add_argument("--classifier_input", choices=['mel32'], default='mel32', help='input of NN')
    parser.add_argument("--num_per_class", type=int, default=10)

    '''DiffWave arguments'''
    parser.add_argument('--config', type=str, default='configs/config.json', help='JSON file for configuration')
    parser.add_argument('--defender_path', type=str, help='dir of diffusion model checkpoint')

    '''certified robustness arguments'''
    parser.add_argument('--defense_method', type=str, default='diffusion', choices=['diffusion', 'randsmooth'])
    parser.add_argument('--sigma', type=float, default=0.25)
    parser.add_argument('--num_sampling', type=int, default=1000)

    '''device arguments'''
    parser.add_argument("--dataload_workers_nums", type=int, default=8, help='number of workers for dataloader')
    parser.add_argument("--batch_size", type=int, default=10, help='batch size')
    parser.add_argument('--gpu', type=int, default=0)

    '''file saving arguments'''
    parser.add_argument('--save_path', type=str, default='_Experiments/certified_robustness/records')

    args = parser.parse_args()


    '''device setting'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)
    print('gpu id: {}'.format(args.gpu))

    '''SC09 classifier setting'''
    #region
    from transforms import *
    from datasets.sc_dataset import *
    from audio_models.create_model import *

    Classifier = create_model(args.classifier_path)
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        Classifier.cuda()

    transform = Compose([LoadAudio(), FixAudioLength()])
    test_dataset = SC09Dataset(folder=args.data_path, transform=transform, num_per_class=args.num_per_class)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=None, shuffle=False, 
                                pin_memory=use_gpu, num_workers=args.dataload_workers_nums)
    criterion = torch.nn.CrossEntropyLoss()
    #endregion

    '''DiffWave denoiser setting'''
    #region
    from diffusion_models.diffwave_ddpm import create_diffwave_model
    DiffWave_Denoiser = create_diffwave_model(model_path=args.defender_path, config_path=args.config)
    DiffWave_Denoiser.eval().cuda()
    #endregion

    '''preprocessing setting'''
    #region
    n_mels = 32
    if args.classifier_input == 'mel40':
        n_mels = 40
    MelSpecTrans = torchaudio.transforms.MelSpectrogram(n_fft=2048, hop_length=512, n_mels=n_mels, norm='slaney', pad_mode='constant', mel_scale='slaney')
    Amp2DB = torchaudio.transforms.AmplitudeToDB(stype='power')
    Wave2Spect = Compose([MelSpecTrans.cuda(), Amp2DB.cuda()])
    #endregion

    '''robust certificate setting'''
    from robustness_eval.certified_robust import *
    if args.defense_method == 'diffusion':
        RC = RobustCertificate(classifier=Classifier, transform=Wave2Spect, denoiser=DiffWave_Denoiser)
    elif args.defense_method == 'randsmooth':
        RC = RobustCertificate(classifier=Classifier, transform=Wave2Spect, denoiser=None)

    '''robustness eval'''
    from tqdm import tqdm
    pbar = tqdm(test_dataloader, unit="audios", unit_scale=test_dataloader.batch_size)

    total = 0

    record_save = []

    for batch in pbar:
        
        waveforms = batch['samples']
        waveforms = torch.unsqueeze(waveforms, 1)
        targets = batch['target']
        

        waveforms = waveforms.cuda()
        targets = targets.cuda()


        '''certified robust accuracy'''
        y_certified, r_certified = RC.certify(x=waveforms, y=targets, 
                                            sigma=args.sigma, n_0=100, n=args.num_sampling, 
                                            batch_size=args.batch_size)

        for i in range(waveforms.shape[0]):
            save_dict = {'id': i+total, 
                         'y_true': targets[i].item(), 
                         'y_pred': y_certified[i].item(), 
                         'certified_radius': r_certified[i].item()}
            record_save.append(save_dict)

        total += waveforms.shape[0]

        save_path = os.path.join(args.save_path, 'sigma={}'.format(args.sigma))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        f = open(os.path.join(save_path, 'sigma={}_N={}.json'.format(args.sigma, args.num_sampling)), 'w')
        json.dump(record_save, f, indent=4)
        f.close()

