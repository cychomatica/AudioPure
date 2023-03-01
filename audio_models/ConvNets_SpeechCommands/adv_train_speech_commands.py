'''Train a CNN for Google speech commands.'''
'''https://github.com/tugstugi/pytorch-speech-commands'''

# import sys
# sys.path.insert(0, './audio_models/ConvNets_SpeechCommands')

import os

import argparse
import time

from tqdm import *

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from torchvision.transforms import *

import models
from .datasets import *
from .transforms import *
from .mixup import *
from typing import Union

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--train_dataset", type=str, help='path of train dataset')
parser.add_argument("--valid_dataset", type=str, help='path of validation dataset')
parser.add_argument("--background_noise", type=str, help='path of background noise')

parser.add_argument("--batch_size", type=int, default=96, help='batch size')
parser.add_argument("--dataload_workers_nums", type=int, default=4, help='number of workers for dataloader')
parser.add_argument("--weight_decay", type=float, default=1e-2, help='weight decay')
parser.add_argument("--optim", choices=['sgd', 'adam'], default='sgd', help='choices of optimization algorithms')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='learning rate for optimization')
parser.add_argument("--lr_scheduler", choices=['plateau', 'step'], default='plateau', help='method to adjust learning rate')
parser.add_argument("--lr_scheduler_patience", type=int, default=5, help='lr scheduler plateau: Number of epochs with no improvement after which learning rate will be reduced')
parser.add_argument("--lr_scheduler_step_size", type=int, default=50, help='lr scheduler step: number of epochs of learning rate decay.')
parser.add_argument("--lr_scheduler_gamma", type=float, default=0.1, help='learning rate is multiplied by the gamma to decrease it')
parser.add_argument("--max_epochs", type=int, default=70, help='max number of epochs')
parser.add_argument('--eps', type=float, default=0.01)
parser.add_argument('--adv_ratio', type=float, default=0.5)

parser.add_argument("--model", choices=models.available_models, default=models.available_models[9], help='model of NN')
parser.add_argument("--input", choices=['mel32'], default='mel32', help='input of NN')
parser.add_argument('--mixup', action='store_true', help='use mixup')
parser.add_argument('--save_path', type=str, help='path of saved model')
parser.add_argument('--gpu_id', type=int, default=0)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
use_gpu = torch.cuda.is_available()
print('use_gpu', use_gpu)
if use_gpu:
    torch.backends.cudnn.benchmark = True

import torchaudio
n_mels = 32
if args.input == 'mel40':
    n_mels = 40
MelSpecTrans = torchaudio.transforms.MelSpectrogram(n_fft=2048, hop_length=512, n_mels=n_mels, norm='slaney', pad_mode='constant', mel_scale='slaney')
Amp2DB = torchaudio.transforms.AmplitudeToDB(stype='power')
Wave2Spect = Compose([MelSpecTrans.cuda(), Amp2DB.cuda()])

transform = Compose([LoadAudio(), FixAudioLength()])    
train_dataset = SpeechCommandsDataset(args.train_dataset,transform)    
valid_dataset = SpeechCommandsDataset(args.valid_dataset,transform)                                 

weights = train_dataset.make_weights_for_balanced_classes()
sampler = WeightedRandomSampler(weights, len(weights))
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                              pin_memory=use_gpu, num_workers=args.dataload_workers_nums)
valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              pin_memory=use_gpu, num_workers=args.dataload_workers_nums)


# a name used to save checkpoints etc.
full_name = '%s_%s_%s_bs%d_lr%.1e_wd%.1e' % (args.model, args.optim, args.lr_scheduler, args.batch_size, args.learning_rate, args.weight_decay)
if args.comment:
    full_name = '%s_%s' % (full_name, args.comment)

model = models.create_model(model_name=args.model, num_classes=len(CLASSES), in_channels=1)

if use_gpu:
    model = torch.nn.DataParallel(model).cuda()

criterion = torch.nn.CrossEntropyLoss()

if args.optim == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

start_timestamp = int(time.time()*1000)
start_epoch = 0
best_accuracy = 0
best_loss = 1e100
global_step = 0


if args.lr_scheduler == 'plateau':
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_scheduler_patience, factor=args.lr_scheduler_gamma)
else:
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step_size, gamma=args.lr_scheduler_gamma, last_epoch=start_epoch-1)

def get_lr():
    return optimizer.param_groups[0]['lr']


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

def pgd(model: torch.nn.Module, 
        x: torch.Tensor, y: torch.Tensor, 
        eps: float=0.002, alpha: float=0.0004, 
        n: int=10, p='linf'):

    assert x.max() <= 1 and x.min() >= -1

    delta = torch.rand_like(x, requires_grad=True).to(x.device)
    delta.data = eps * (2 * delta.data - 1)
    delta.data = (x + delta.data).clamp(-1, 1) - x

    batch_size = x.shape[0]
    x_adv = [None] * batch_size

    for i in range(n+1):

        x_pert = x + delta
        y_pert = model(Wave2Spect(x_pert))

        for j in range(batch_size):
            if y[j] != y_pert[j].max(0, keepdim=True)[1]:
                x_adv[j] = x_pert[j]
            
        if i == n: break

        loss = torch.nn.CrossEntropyLoss()(y_pert, y)
        loss.backward()

        if p == 'linf':
            delta.data = (delta.data + alpha * delta.grad.data.sign()).clamp(-eps, eps)
            delta.data = (x + delta.data).clamp(-1, 1) - x
        elif p == 'l2':
            delta.data = project_to_norm_ball(delta.data + alpha * project_to_norm_ball(delta.grad.data, p='l2', eps=1), p='l2', eps=eps)
            delta.data = (x + delta.data).clamp(-1, 1) - x
        else:
            raise NotImplementedError(f'Unsupported norm: {p}!')
        delta.grad.zero_()
    
    for j in range(batch_size):
        if x_adv[j] is None:
            x_adv[j] = x_pert[j]
    
    x_adv = torch.unsqueeze(torch.cat(x_adv, dim=0), 1)

    return x_adv


def train(epoch):
    global global_step

    print("epoch %3d with lr=%.02e" % (epoch, get_lr()))
    phase = 'train'
    model.train()  # Set model to training mode

    running_loss = 0.0
    it = 0
    correct = 0
    correct_adv = 0
    total = 0

    pbar = tqdm(train_dataloader, unit="audios", unit_scale=train_dataloader.batch_size)
    for batch in pbar:
        inputs = batch['samples']
        inputs = torch.unsqueeze(inputs, 1)
        targets = batch['target']

        if args.mixup:
            inputs, targets = mixup(inputs, targets, num_classes=len(CLASSES))

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda(non_blocking=True)
        
        nb_adv = int(np.ceil(args.adv_ratio * inputs.shape[0]))
        if args.adv_ratio < 1:
            adv_ids = np.random.choice(inputs.shape[0], size=nb_adv, replace=False)
        else:
            adv_ids = list(range(inputs.shape[0])) 
            np.random.shuffle(adv_ids)

        inputs[adv_ids] = pgd(model, inputs[adv_ids].clone(), targets[adv_ids].clone(), 
                              eps=args.eps, alpha=args.eps/5, n=10, p='linf')
        outputs = model(Wave2Spect(inputs))
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        it += 1
        global_step += 1
        total += targets.size(0)
        running_loss += loss.item()
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum()

        # update the progress bar
        pbar.set_postfix({
            'loss': "%.05f" % (running_loss / it),
            'acc': "%.02f%%" % (100*correct/total), 
            'robust acc': "%.02f%%" % (100*correct_adv/total), 
            'total acc': "%.02f%%" % (50*(correct+correct_adv)/total), 
        })

    accuracy = correct/total
    epoch_loss = running_loss / it

    print('\nepoch {}'.format(epoch))
    print('train_accuracy: {}\t train_loss: {}\n'.format(accuracy, epoch_loss))

def valid(epoch):
    global best_accuracy, best_loss, global_step

    phase = 'valid'
    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    it = 0
    correct = 0
    total = 0

    pbar = tqdm(valid_dataloader, unit="audios", unit_scale=valid_dataloader.batch_size)
    for batch in pbar:
        inputs = batch['samples']
        inputs = torch.unsqueeze(inputs, 1)
        targets = batch['target']

        inputs = Variable(inputs, requires_grad=False)
        targets = Variable(targets, requires_grad=False)

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda(non_blocking=True)
        
        inputs = pgd(model, inputs, targets, 
                     eps=args.eps, alpha=args.eps/5, n=10, p='linf')

        # forward
        outputs = model(Wave2Spect(inputs))
        loss = criterion(outputs, targets)

        # statistics
        it += 1
        global_step += 1
        running_loss += loss.item()
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum()
        total += targets.size(0)

        # update the progress bar
        pbar.set_postfix({
            'loss': "%.05f" % (running_loss / it),
            'acc': "%.02f%%" % (100*correct/total)
        })

    accuracy = correct/total
    epoch_loss = running_loss / it

    print('\nepoch {}'.format(epoch))
    print('val_robust_accuracy: {}\t val_robust_loss: {}\n'.format(accuracy, epoch_loss))

    save_path = '{}/{}'.format(args.save_path, full_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model, os.path.join(save_path, 'audio-advtr-ratio={}-linf={}-best-robust-loss.pth'.format(args.adv_ratio, args.eps)))
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model, os.path.join(save_path, 'audio-advtr-ratio={}-linf={}-best-robust-acc.pth'.format(args.adv_ratio, args.eps)))

    return epoch_loss

print("training %s for Google speech commands..." % args.model)
since = time.time()
for epoch in range(start_epoch, args.max_epochs):
    if args.lr_scheduler == 'step':
        lr_scheduler.step()

    train(epoch)
    epoch_loss = valid(epoch)

    if args.lr_scheduler == 'plateau':
        lr_scheduler.step(metrics=epoch_loss)

    time_elapsed = time.time() - since
    time_str = 'total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600, time_elapsed % 3600 // 60, time_elapsed % 60)
    print("%s, best robust accuracy: %.02f%%, best robust loss %f" % (time_str, 100*best_accuracy, best_loss))
print("finished")
