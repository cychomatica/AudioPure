import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchaudio
from tqdm import tqdm
import os
import numpy as np

from qualcomm_kws_dataset import QualcommKeywordSpottingDataset, LoadAudio
from model import KWSModel

BATCH_SIZE = 128
NUM_EPOCHS = 1
N_MELS     = 32

IN_SIZE = N_MELS
HIDDEN_SIZE = 64
KERNEL_SIZE = (20, 5)
STRIDE = (8, 2)
GRU_NUM_LAYERS = 2
NUM_DIRS = 2
NUM_CLASSES = 4

DATA_PATH = '' # qualcomm keyword speech dataset folders
SAVE_PATH = '' # saving dir

def set_audio_length(data, time):
    
    samples = data['samples']
    sample_rate = data['sample_rate']
    length = int(time * sample_rate)
    if length < len(samples):
        data['samples'] = samples[:length]
    elif length > len(samples):
        data['samples'] = np.pad(samples, (0, length - len(samples)), "constant")
    return data

def rand_audio_length_collate(batch):

    time = np.random.uniform(0.74751, 2.74751)
    batch = [set_audio_length(data, time) for data in batch]

    path = [data['path'] for data in batch]
    sample_rate = torch.tensor([data['sample_rate'] for data in batch])
    samples = torch.cat([torch.from_numpy(data['samples']).unsqueeze(0) for data in batch], dim=0)
    target = torch.tensor([data['target'] for data in batch])

    return {'path': path, 'sample_rate': sample_rate, 'samples': samples, 'target': target}

from torchvision.transforms.transforms import Compose
TrainData = QualcommKeywordSpottingDataset(folder=DATA_PATH, usage='Train', transform=LoadAudio())
ValidData = QualcommKeywordSpottingDataset(folder=DATA_PATH, usage='Valid', transform=LoadAudio())
TestData = QualcommKeywordSpottingDataset(folder=DATA_PATH, usage='Test', transform=LoadAudio())
TrainLoader = DataLoader(TrainData, batch_size=BATCH_SIZE, shuffle=True, collate_fn=rand_audio_length_collate, num_workers=8)
ValidLoader = DataLoader(ValidData, batch_size=1, shuffle=False, num_workers=0)
TestLoader = DataLoader(TestData, batch_size=1, shuffle=False, num_workers=0)


All_Data = QualcommKeywordSpottingDataset(folder=DATA_PATH, usage='All', transform=LoadAudio())
All_Loader = DataLoader(All_Data, batch_size=1, shuffle=False, num_workers=0)
time_max, time_min = 16000, 16000
i = 0
for batch in All_Loader:
    audio = batch['samples']
    if audio.shape[-1] > time_max:
        time_max = audio.shape[-1]
    if audio.shape[-1] < time_min:
        time_min = audio.shape[-1]
    i = i + 1
print('max duration: ', time_max / 16000)
print('min duration: ', time_min / 16000)

MelSpecTrans = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=N_MELS)
Amp2DB = torchaudio.transforms.AmplitudeToDB(stype='power')
Wave2Spect = nn.Sequential(MelSpecTrans.cuda(), Amp2DB.cuda())

KWS = KWSModel(IN_SIZE, HIDDEN_SIZE, KERNEL_SIZE, STRIDE, GRU_NUM_LAYERS, NUM_DIRS, NUM_CLASSES)
KWS.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(KWS.parameters(), weight_decay=1e-5)
adv = True

'''pgd attack'''
def pgd(model: torch.nn.Module, transform, 
        x: torch.Tensor, y: torch.Tensor, 
        criterion: torch.nn.modules.loss._Loss, 
        eps: float=0.002, alpha: float=0.0004, n: int=10):

    assert x.max() <= 1 and x.min() >= -1

    delta = torch.rand_like(x, requires_grad=True).to(x.device)
    delta.data = eps * (2 * delta.data - 1)
    delta.data = (x + delta.data).clamp(-1, 1) - x

    batch_size = x.shape[0]
    x_adv = [None] * batch_size

    for i in range(n+1):

        x_pert = x + delta
        y_pert = model(transform(x_pert))

        for j in range(batch_size):
            if y[j] != y_pert[j].max(0, keepdim=True)[1]:
                x_adv[j] = x_pert[j]
            
        if i == n: break

        loss = criterion(y_pert, y)
        loss.backward()

        delta.data = (delta.data + alpha * delta.grad.data.sign()).clamp(-eps, eps)
        delta.data = (x + delta.data).clamp(-1, 1) - x
    
    for j in range(batch_size):
        if x_adv[j] is None:
            x_adv[j] = x_pert[j]
    
    x_adv = torch.unsqueeze(torch.cat(x_adv, dim=0), 1)

    return x_adv

'''training'''
def train(epoch, trainloader, adv_train=False):
    train_step, train_loss, train_correct, train_total = 0, 0., 0, 0
    pbar_train = tqdm(trainloader, unit="audio samples", unit_scale=trainloader.batch_size)
    # pbar_train.iterable.dataset.transform.transforms[1].time = np.random.uniform(1, 2)

    KWS.train()

    for batch in pbar_train:
        
        train_step += 1

        waveforms = batch['samples']
        waveforms = torch.unsqueeze(waveforms, 1)
        targets = batch['target']

        waveforms = waveforms.cuda()
        targets = targets.cuda()

        if adv_train:
            inputs = pgd(KWS, Wave2Spect, x=waveforms, y=targets, criterion=criterion)
        else:
            inputs = waveforms

        spectrogram = Wave2Spect(inputs)
        outputs = KWS(spectrogram)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = outputs.data.max(1, keepdim=True)[1]
        train_correct += pred.eq(targets.data.view_as(pred)).sum()
        train_total += targets.size(0)
        
        pbar_train.set_postfix({
                                'train_loss': "%.05f" % (train_loss / train_step),
                                'acc': "%.02f%%" % (100*train_correct / train_total)
                                })
    accuracy = train_correct / train_total
    epoch_loss = train_loss / train_step
    print('epoch {}'.format(epoch))
    print('train_accuracy: {}\t train_loss: {}\n'.format(accuracy, epoch_loss))


'''validation'''
def valid(epoch, testloader, adv_eval=False):
    global best_valid_clean_acc, best_valid_robust_acc

    test_step, test_loss, test_correct, test_total = 0, 0., 0, 0
    pbar_test = tqdm(testloader, unit="audio samples", unit_scale=testloader.batch_size)

    # KWS.eval()

    for batch in pbar_test:

        waveforms = batch['samples']
        waveforms = torch.unsqueeze(waveforms, 1)
        targets = batch['target']

        waveforms = waveforms.cuda()
        targets = targets.cuda()

        if adv_eval:
            inputs = pgd(KWS, Wave2Spect, x=waveforms, y=targets, criterion=criterion)
        else:
            inputs = waveforms

        spectrogram = Wave2Spect(inputs)
        outputs = KWS(spectrogram)
        # outputs = outputs.unsqueeze(0)
        loss = criterion(outputs, targets)

        test_step += 1
        test_loss += loss.item()
        pred = outputs.data.max(1, keepdim=True)[1]
        test_correct += pred.eq(targets.data.view_as(pred)).sum()
        test_total += targets.size(0)

        info_out = {
                    'test_loss': "%.05f" % (test_loss / test_step),
                    'acc': "%.02f%%" % (100*test_correct / test_total)
                    }
        pbar_test.set_postfix(info_out)

    accuracy = test_correct / test_total
    epoch_loss = test_loss / test_step

    if adv_eval:
        print('epoch {}'.format(epoch))
        print('robust test accuracy: {}\t robust test loss: {}\n'.format(accuracy, epoch_loss))

        if accuracy > best_valid_robust_acc:
            best_valid_robust_acc = accuracy
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            torch.save(KWS.state_dict(), os.path.join(SAVE_PATH, 'advtr-best-robust-acc-kws-attn_rcnn-n_mels={}.pth'.format(N_MELS)))
            print('checkpoint saved.')
    
    else:
        print('epoch {}'.format(epoch))
        print('clean test accuracy: {}\t clean test loss: {}\n'.format(accuracy, epoch_loss))

        if accuracy > best_valid_clean_acc:
            best_valid_clean_acc = accuracy
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            torch.save(KWS.state_dict(), os.path.join(SAVE_PATH, 'advtr-best-clean-acc-kws-attn_rcnn-n_mels={}.pth'.format(N_MELS)))
            print('checkpoint saved.')

best_valid_clean_acc = 0
best_valid_robust_acc = 0
epoch = 100
for ep in range(1, epoch+1):
    train(ep, TrainLoader, adv_train=True)
    valid(ep, ValidLoader, adv_eval=True)
    valid(ep, ValidLoader, adv_eval=False)

KWS.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'advtr-best-robust-acc-kws-attn_rcnn-n_mels={}.pth'.format(N_MELS))))
valid('valid acc best ckpt', TestLoader, adv_eval=True)
valid('valid acc best ckpt', TestLoader, adv_eval=False)