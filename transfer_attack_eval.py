import os
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import *
import torchaudio

from diffusion_models.diffwave_ddpm import DiffWave
from robustness_eval.black_box_attack import FAKEBOB

import utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    '''SC09 classifier arguments'''
    parser.add_argument("--data_path", default='datasets/speech_commands/test')
    parser.add_argument("--classifier_path", help='dir of saved classifier model')
    parser.add_argument("--classifier_input", choices=['mel32'], default='mel32', help='input of NN')
    parser.add_argument("--num_per_class", type=int, default=10)

    '''DiffWave-VPSDE arguments'''
    parser.add_argument('--ddpm_config', type=str, default='configs/config.json', help='JSON file for configuration')
    parser.add_argument('--ddpm_path', type=str, help='dir of diffusion model checkpoint')
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=3, help='diffusion steps, control the sampling noise scale')
    parser.add_argument('--t_delta', type=int, default=0, help='perturbation range of sampling noise scale; set to 0 by default')
    parser.add_argument('--rand_t', action='store_true', default=False, help='decide if randomize sampling noise scale')
    parser.add_argument('--diffusion_type', type=str, default='ddpm', help='[ddpm, sde]')
    parser.add_argument('--score_type', type=str, default='guided_diffusion', help='[guided_diffusion, score_sde, ddpm]')
    parser.add_argument('--use_bm', action='store_true', default=False, help='whether to use brownian motion')

    '''attack arguments'''
    parser.add_argument("--surrogate_path", help='dir of saved surrogate model')
    parser.add_argument('--attack', type=str, choices=['PGD', 'FAKEBOB'], default='PGD')
    parser.add_argument('--defense', type=str, choices=['Diffusion', 'Diffusion-Spec', 'AS', 'MS', 'DS', 'LPF', 'BPF', 'None'], default='Diffusion')
    parser.add_argument('--bound_norm', type=str, choices=['linf', 'l2'], default='linf')
    parser.add_argument('--eps', type=int, default=0.002)
    parser.add_argument('--max_iter_1', type=int, default=70)
    parser.add_argument('--max_iter_2', type=int, default=0)
    parser.add_argument('--eot_attack_size', type=int, default=1, help='EOT size of attack')
    parser.add_argument('--eot_defense_size', type=int, default=1)
    parser.add_argument('--verbose', type=int, default=0)

    '''device arguments'''
    parser.add_argument("--dataload_workers_nums", type=int, default=8, help='number of workers for dataloader')
    parser.add_argument("--batch_size", type=int, default=10, help='batch size')
    parser.add_argument('--gpu', type=int, default=0)

    '''file saving arguments'''
    parser.add_argument('--save_path', default=None)

    args = parser.parse_args()


    '''device setting'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)
    print('gpu id: {}'.format(args.gpu))


    '''SC09 classifier setting'''
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

    '''preprocessing setting (if use acoustic features like mel-spectrogram)'''
    n_mels = 32
    if args.classifier_input == 'mel40':
        n_mels = 40
    MelSpecTrans = torchaudio.transforms.MelSpectrogram(n_fft=2048, hop_length=512, n_mels=n_mels, norm='slaney', pad_mode='constant', mel_scale='slaney')
    Amp2DB = torchaudio.transforms.AmplitudeToDB(stype='power')
    Wave2Spect = Compose([MelSpecTrans.cuda(), Amp2DB.cuda()])

    '''defense setting'''
    from acoustic_system import AcousticSystem
    if args.defense == 'None':
        if Classifier._get_name() == 'M5': # M5Net takes the raw audio as input
            AS_MODEL = AcousticSystem(classifier=Classifier, transform=None, defender=None)
        else: 
            AS_MODEL = AcousticSystem(classifier=Classifier, transform=Wave2Spect, defender=None)
        print('classifier model: {}'.format(Classifier._get_name()))
        print('defense: None')

    else:
        if args.defense == 'Diffusion':
            from diffusion_models.diffwave_sde import *
            Defender = RevDiffWave(args)
            defense_type = 'wave'
        elif args.defense == 'Diffusion-Spec':
            from diffusion_models.improved_diffusion_sde import *
            Defender = RevImprovedDiffusion(args)
            defense_type = 'spec'
        elif args.defense == 'AS': 
            from transforms.time_defense import *
            Defender = TimeDomainDefense(defense_type='AS')
            defense_type = 'wave'
        elif args.defense == 'MS': 
            from transforms.time_defense import *
            Defender = TimeDomainDefense(defense_type='MS')
            defense_type = 'wave'
        elif args.defense == 'DS': 
            from transforms.frequency_defense import *
            Defender = FreqDomainDefense(defense_type='DS')
            defense_type = 'wave'
        elif args.defense == 'LPF': 
            from transforms.frequency_defense import *
            Defender = FreqDomainDefense(defense_type='LPF')
            defense_type = 'wave'
        elif args.defense == 'BPF': 
            from transforms.frequency_defense import *
            Defender = FreqDomainDefense(defense_type='BPF')
            defense_type = 'wave'
        else:
            raise NotImplementedError(f'Unknown defense: {args.defense}!')
        
        if Classifier._get_name() == 'M5':
            AS_MODEL = AcousticSystem(classifier=Classifier, transform=None, defender=Defender, defense_type=defense_type)
        else: 
            AS_MODEL = AcousticSystem(classifier=Classifier, transform=Wave2Spect, defender=Defender, defense_type=defense_type)
        
        print('classifier model: {}'.format(Classifier._get_name()))
        if args.defense == 'Diffusion':
            print('defense: {} with t={}'.format(Defender._get_name(), args.t))
        else:
            print('defense: {}'.format(Defender._get_name()))
    AS_MODEL.eval()


    Surrogate = create_model(args.surrogate_path)
    if use_gpu:
        torch.backends.cudnn.benchmark = True
    Surrogate.cuda()
    if Surrogate._get_name() == 'M5': # M5Net takes the raw audio as input
        SUR_MODEL = AcousticSystem(classifier=Surrogate, transform=None, defender=None)
    else: 
        SUR_MODEL = AcousticSystem(classifier=Surrogate, transform=Wave2Spect, defender=None)
    print('surrogate model: {}'.format(Classifier._get_name()))
    print('defense: None')
    SUR_MODEL.eval()

    '''attack setting'''
    from robustness_eval.white_box_attack import *
    if args.attack == 'PGD':
        Attacker = AudioAttack(model=SUR_MODEL, 
                                eps=args.eps, norm=args.bound_norm,
                                max_iter_1=args.max_iter_1, max_iter_2=0,
                                learning_rate_1=args.eps/5 if args.bound_norm=='linf' else args.eps/50, 
                                eot_attack_size=args.eot_attack_size,
                                eot_defense_size=args.eot_defense_size,
                                verbose=args.verbose)
        print('attack: {} with {}_eps={} & iter={} & eot={}-{}\n'\
            .format(args.attack, args.bound_norm, args.eps, args.max_iter_1, args.eot_attack_size, args.eot_defense_size))
    elif args.attack == 'FAKEBOB':
        eps = args.eps #/ (2**15)
        confidence = 0.5
        max_iter = 200
        samples_per_draw = 200
        samples_per_draw_batch_size = 50
        Attacker = FAKEBOB(model=SUR_MODEL, task='SCR', targeted=False, verbose=args.verbose,
                           confidence=confidence, epsilon=eps, max_lr=0.001, min_lr=1e-6,
                           max_iter=max_iter, samples_per_draw=samples_per_draw, samples_per_draw_batch_size=samples_per_draw_batch_size, batch_size=args.batch_size)
        print('attack: {} with eps={} & confidence={} & iter={} & samples_per_draw={}\n'\
            .format(args.attack, eps, confidence, max_iter, samples_per_draw))
    else:
        raise AttributeError("this version does not support '{}' at present".format(args.attack))


    '''robustness eval'''
    from tqdm import tqdm
    pbar = tqdm(test_dataloader, unit="audios", unit_scale=test_dataloader.batch_size)

    correct_steal = 0
    correct_vanilla_clean = 0
    correct_vanilla_robust = 0
    correct_defended_clean = 0
    correct_defended_robust = 0
    total = 0

    for batch in pbar:
        
        waveforms = batch['samples']
        waveforms = torch.unsqueeze(waveforms, 1)
        targets = batch['target']

        waveforms = waveforms.cuda()
        targets = targets.cuda()

        '''original audio'''
        pred_vanilla_clean = AS_MODEL(waveforms, False).max(1, keepdim=True)[1].squeeze()
        pred_defended_clean = AS_MODEL(waveforms, True).max(1, keepdim=True)[1].squeeze()
        pred_steal = SUR_MODEL(waveforms, False).max(1, keepdim=True)[1].squeeze()

        '''adversarial audio'''
        waveforms_adv, _ = Attacker.generate(x=waveforms, y=targets, targeted=False)
        if isinstance(waveforms_adv, np.ndarray):
            if waveforms_adv.dtype == np.int16 and waveforms_adv.max() > 1 and waveforms_adv.min() < -1:
                waveforms_adv = waveforms_adv / (2**15)
            waveforms_adv = torch.tensor(waveforms_adv, dtype=waveforms.dtype).to(waveforms.device)
        
        pred_vanilla_adv = AS_MODEL(waveforms_adv, False).max(1, keepdim=True)[1].squeeze()
        pred_defended_adv = AS_MODEL(waveforms_adv, True).max(1, keepdim=True)[1].squeeze()


        '''metrics output'''
        total += waveforms.shape[0]
        correct_steal += (pred_steal==pred_defended_clean).sum().item()
        correct_vanilla_clean += (pred_vanilla_clean==targets).sum().item()
        correct_defended_clean += (pred_defended_clean==targets).sum().item()
        correct_vanilla_robust += (pred_vanilla_adv==targets).sum().item()
        correct_defended_robust += (pred_defended_adv==targets).sum().item()

        acc_steal = correct_steal / total * 100
        acc_vanilla_clean = correct_vanilla_clean / total * 100
        acc_defended_clean = correct_defended_clean / total * 100
        acc_vanilla_robust = correct_vanilla_robust / total * 100
        acc_defended_robust = correct_defended_robust / total * 100

        pbar_info = {
                    'acc_steal: ': '{:.4f}%'.format(acc_steal),
                    'acc_vanilla_clean: ': '{:.4f}%'.format(acc_vanilla_clean),
                    'acc_defended_clean: ': '{:.4f}%'.format(acc_defended_clean),
                    'acc_vanilla_robust: ': '{:.4f}%'.format(acc_vanilla_robust),
                    'acc_defended_robust: ': '{:.4f}%'.format(acc_defended_robust)
                    }

        pbar.set_postfix(pbar_info)
        pbar.update(1)


    '''summary'''
    print('on {} test examples: '.format(total))
    print('acc_steal: {:.4f}%'.format(acc_steal))
    print('acc_vanilla_clean: {:.4f}%'.format(acc_vanilla_clean))
    print('acc_defended_clean: {:.4f}%'.format(acc_defended_clean))
    print('acc_vanilla_robust: {:.4f}%'.format(acc_vanilla_robust))
    print('acc_defended_robust: {:.4f}%'.format(acc_defended_robust))
