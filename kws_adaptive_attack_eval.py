import os
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import *
import torchaudio

from robustness_eval.black_box_attack import FAKEBOB
import utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    '''Qualcomm Keyword Spotting classifier arguments'''
    parser.add_argument("--data_path", help='qualcomm keyword speech dataset folder')
    parser.add_argument("--classifier_path", help='dir of model checkpoint')
    parser.add_argument("--classifier_input", choices=['mel32'], default='mel32', help='input of NN')
    parser.add_argument("--num_per_class", type=int, default=10)

    '''DiffWave-VPSDE arguments'''
    parser.add_argument('--ddpm_config', type=str, default='configs/config.json', help='JSON file for configuration')
    parser.add_argument('--ddpm_path', type=str, help='dir of diffusion model checkpoint')
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=2, help='diffusion steps, control the sampling noise scale')
    parser.add_argument('--t_delta', type=int, default=0, help='perturbation range of sampling noise scale; set to 0 by default')
    parser.add_argument('--rand_t', action='store_true', default=False, help='decide if randomize sampling noise scale')
    parser.add_argument('--diffusion_type', type=str, default='ddpm', help='[ddpm, sde]')
    parser.add_argument('--score_type', type=str, default='guided_diffusion', help='[guided_diffusion, score_sde, ddpm]')
    parser.add_argument('--use_bm', action='store_true', default=False, help='whether to use brownian motion')

    '''attack arguments'''
    parser.add_argument('--attack', type=str, choices=['PGD', 'FAKEBOB'], default='PGD')
    parser.add_argument('--defense', type=str, choices=['Diffusion', 'DiffSpec', 'AS', 'MS', 'DS', 'LPF', 'BPF', 'None'], default='None')
    parser.add_argument('--bound_norm', type=str, choices=['linf', 'l2'], default='linf')
    parser.add_argument('--eps', type=float, default=0.002)
    parser.add_argument('--max_iter_1', type=int, default=70)
    parser.add_argument('--max_iter_2', type=int, default=0)
    parser.add_argument('--eot_attack_size', type=int, default=1, help='EOT size of attack')
    parser.add_argument('--eot_defense_size', type=int, default=1)
    parser.add_argument('--verbose', type=int, default=0)

    '''device arguments'''
    parser.add_argument("--dataload_workers_nums", type=int, default=0, help='number of workers for dataloader')
    parser.add_argument("--batch_size", type=int, default=1, help='batch size')
    parser.add_argument('--gpu', type=int, default=1)

    '''file saving arguments'''
    parser.add_argument('--save_path', default=None)

    args = parser.parse_args()

    '''device setting'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)
    print('gpu id: {}'.format(args.gpu))


    '''preprocessing setting (if use acoustic features like mel-spectrogram)'''
    n_mels = 32
    if args.classifier_input == 'mel40':
        n_mels = 40
    MelSpecTrans = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=n_mels)
    Amp2DB = torchaudio.transforms.AmplitudeToDB(stype='power')
    Wave2Spect = torch.nn.Sequential(MelSpecTrans.cuda(), Amp2DB.cuda())

    '''QKWS classifier setting'''
    from transforms import *
    from audio_models.RCNN_KWS import *

    Classifier = KWSModel(in_size=n_mels)
    Classifier.load_state_dict(torch.load(args.classifier_path))
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        Classifier.cuda()

    test_dataset = QualcommKeywordSpottingDataset(folder=args.data_path, usage='Test', transform=LoadAudio())
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    criterion = torch.nn.CrossEntropyLoss()

    '''defense setting'''
    from acoustic_system import AcousticSystem
    if args.defense == 'None':

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

        AS_MODEL = AcousticSystem(classifier=Classifier, transform=Wave2Spect, defender=Defender, defense_type=defense_type)
        print('classifier model: {}'.format(Classifier._get_name()))
        if args.defense == 'Diffusion':
            print('defense: {} with t={}'.format(Defender._get_name(), args.t))
        else:
            print('defense: {}'.format(Defender._get_name()))
    AS_MODEL.eval()


    '''attack setting'''
    from robustness_eval.white_box_attack import *
    from robustness_eval.white_box_attack import *
    if args.attack == 'PGD':
        Attacker = AudioAttack(model=AS_MODEL, 
                                eps=args.eps, norm=args.bound_norm,
                                max_iter_1=args.max_iter_1, max_iter_2=0,
                                learning_rate_1=args.eps/5 if args.bound_norm=='linf' else args.eps/2, 
                                eot_attack_size=args.eot_attack_size,
                                eot_defense_size=args.eot_defense_size,
                                verbose=args.verbose)
        print('attack: {} with {}_eps={} & iter={} & eot={}-{}\n'\
            .format(args.attack, args.bound_norm, args.eps, args.max_iter_1, args.eot_attack_size, args.eot_defense_size))
    elif args.attack == 'FAKEBOB':
        eps = args.eps 
        confidence = 0.5
        max_iter = 200
        samples_per_draw = 200
        samples_per_draw_batch_size = 50
        Attacker = FAKEBOB(model=AS_MODEL, task='SCR', targeted=False, verbose=args.verbose,
                           confidence=confidence, epsilon=eps, max_lr=0.001, min_lr=1e-6,
                           max_iter=max_iter, samples_per_draw=samples_per_draw, samples_per_draw_batch_size=samples_per_draw_batch_size, batch_size=args.batch_size)
        print('attack: {} with eps={} & confidence={} & iter={} & samples_per_draw={}\n'\
            .format(args.attack, eps, confidence, max_iter, samples_per_draw))
    else:
        raise AttributeError("this version does not support '{}' at present".format(args.attack))

    
    '''robustness eval'''
    from tqdm import tqdm
    pbar = tqdm(test_dataloader, unit="audios", unit_scale=test_dataloader.batch_size)

    correct_orig = 0
    correct_orig_denoised = 0
    correct_adv_1 = 0
    total = 0

    for batch in pbar:
        
        waveforms = batch['samples']
        waveforms = torch.unsqueeze(waveforms, 1)
        targets = batch['target']

        waveforms = waveforms.cuda()
        targets = targets.cuda()

        AS_MODEL.defender.rev_vpsde.audio_shape = (1, waveforms.shape[-1])

        '''original audio'''
        pred_clean = AS_MODEL(waveforms, False).max(1, keepdim=True)[1].squeeze()

        '''denoised original audio'''
        if AS_MODEL.defense_type == 'wave':
            if args.defense == 'None':
                waveforms_defended = waveforms
            else:
                waveforms_defended = AS_MODEL.defender(waveforms)
            pred_defended = AS_MODEL(waveforms_defended, False).max(1, keepdim=True)[1].squeeze()
        elif AS_MODEL.defense_type == 'spec': 
            spectrogram = AS_MODEL.transform(waveforms)
            if args.defense == 'None':
                spectrogram_defended = spectrogram
            else:
                spectrogram_defended = AS_MODEL.defender(spectrogram)
            pred_defended = AS_MODEL.classifier(spectrogram_defended).max(1, keepdim=True)[1].squeeze()
        y_orig_denoised = torch.squeeze(Classifier(Wave2Spect(waveforms_defended)).max(1, keepdim=True)[1])

        '''adversarial audio'''
        waveforms_adv, attack_success = Attacker.generate(x=waveforms, y=targets, targeted=False)
        if isinstance(waveforms_adv, np.ndarray):
            if waveforms_adv.dtype == np.int16 and waveforms_adv.max() > 1 and waveforms_adv.min() < -1:
                waveforms_adv = waveforms_adv / (2**15)
            waveforms_adv = torch.tensor(waveforms_adv, dtype=waveforms.dtype).to(waveforms.device)
        

        '''denoised adversarial audio'''
        if AS_MODEL.defense_type == 'wave':
            if args.defense == 'None':
                waveforms_adv_defended = waveforms_adv
            else:
                waveforms_adv_defended = AS_MODEL.defender(waveforms_adv)
        elif AS_MODEL.defense_type == 'spec':
            spectrogram_adv = AS_MODEL.transform(waveforms_adv)
            if args.defense == 'None':
                spectrogram_adv_defended = spectrogram_adv
            else:
                spectrogram_adv_defended = AS_MODEL.defender(spectrogram_adv)

        '''waveform/spectrogram saving'''
        if args.save_path is not None:
 
            clean_path = os.path.join(args.save_path,'clean')
            adv_path = os.path.join(args.save_path,'adv')

            if not os.path.exists(clean_path):
                os.makedirs(clean_path)
            if not os.path.exists(adv_path):
                os.makedirs(adv_path)

            for i in range(waveforms.shape[0]):
                
                audio_id = str(total + i).zfill(3)

                if AS_MODEL.defense_type == 'wave': 
                    utils.audio_save(waveforms[i], path=clean_path, 
                                     name='{}_{}_clean.wav'.format(audio_id,targets[i].item()))
                    utils.audio_save(waveforms_defended[i], path=clean_path, 
                                     name='{}_{}_clean_purified.wav'.format(audio_id,targets[i].item()))
                    utils.audio_save(waveforms_adv[i], path=adv_path, 
                                     name='{}_{}_adv.wav'.format(audio_id,targets[i].item()))
                    utils.audio_save(waveforms_adv[i], path=adv_path, 
                                     name='{}_{}_adv_purified.wav'.format(audio_id,targets[i].item()))
                elif AS_MODEL.defense_type == 'spec':
                    utils.spec_save(spectrogram[i], path=clean_path, 
                                    name='{}_{}_clean.png'.format(audio_id,targets[i].item()))
                    utils.spec_save(spectrogram_defended[i], path=clean_path, 
                                    name='{}_{}_clean_purified.png'.format(audio_id,targets[i].item()))
                    utils.spec_save(spectrogram_adv[i], path=adv_path, 
                                    name='{}_{}_adv.png'.format(audio_id,targets[i].item()))
                    utils.spec_save(spectrogram_adv_defended[i], path=adv_path, 
                                    name='{}_{}_adv_purified.png'.format(audio_id,targets[i].item()))


        '''metrics output'''
        total += waveforms.shape[0]
        correct_orig += (pred_clean==targets).sum().item()
        correct_orig_denoised += (pred_defended==targets).sum().item()
        acc_orig = correct_orig / total * 100
        acc_orig_denoised = correct_orig_denoised / total * 100

        if isinstance(attack_success, tuple):
            correct_adv_1 += waveforms.shape[0] - torch.tensor(attack_success[0]).sum().item()
            acc_adv_1 = correct_adv_1 / total * 100
            pbar_info = {
                        'orig clean acc: ': '{:.4f}%'.format(acc_orig),
                        'denoised clean acc: ': '{:.4f}%'.format(acc_orig_denoised),
                        '{} robust acc: '.format(args.attack): '{:.4f}%'.format(acc_adv_1)
                        }
        else:
            correct_adv_1 += waveforms.shape[0] - torch.tensor(attack_success).sum().item()
            acc_adv_1 = correct_adv_1 / total * 100

            pbar_info = {
                        'orig clean acc: ': '{:.4f}%'.format(acc_orig),
                        'denoised clean acc: ': '{:.4f}%'.format(acc_orig_denoised),
                        '{} robust acc: '.format(args.attack): '{:.4f}%'.format(acc_adv_1)
                        }

        pbar.set_postfix(pbar_info)
        pbar.update(1)


    '''summary'''
    print('on {} test examples: '.format(total))
    print('original clean test accuracy: {:.4f}%'.format(acc_orig))
    print('denoised clean test accuracy: {:.4f}%'.format(acc_orig_denoised))
    print('{} robust test accuracy: {:.4f}%'.format(args.attack, acc_adv_1))

