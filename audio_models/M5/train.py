if __name__ == '__main__':

    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', default='datasets/speech_commands/train')
    parser.add_argument('--valid_data_path', default='datasets/speech_commands/valid')
    parser.add_argument('--first_kernel_size', default=80)
    parser.add_argument('--epochs', default=100)
    parser.add_argument('--gpu_id', default=2)
    parser.add_argument('--batch_size', default=256)
    parser.add_argument('--dataload_workers_nums', default=8)
    parser.add_argument('--save_path', default='audio_models/M5/checkpoints')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print('gpu_id', args.gpu_id)

    import torch
    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        
    
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import WeightedRandomSampler

    from M5Net import *
    from transforms import *
    from torchvision.transforms import *
    from datasets.sc_dataset import *

    from torch import optim
    from tqdm import tqdm

    transform = Compose([LoadAudio(), FixAudioLength()])
    train_dataset = SC09Dataset(folder=args.train_data_path, transform=transform, num_per_class=74751)
    valid_dataset = SC09Dataset(folder=args.valid_data_path, transform=transform, num_per_class=74751)

    weights = train_dataset.make_weights_for_balanced_classes()
    sampler = WeightedRandomSampler(weights, len(weights))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                                pin_memory=use_gpu, num_workers=args.dataload_workers_nums)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                pin_memory=use_gpu, num_workers=args.dataload_workers_nums)

    model = M5(n_output=10, first_kernel_size=args.first_kernel_size)

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_valid_accuracy = 0.

    def train(model, epoch):

        running_loss = 0.0
        iteration = 0
        correct = 0
        total = 0

        model.train()
        pbar = tqdm(train_loader, unit="audios", unit_scale=train_loader.batch_size)

        for batch in pbar:
            
            waveforms = batch['samples']
            waveforms = torch.unsqueeze(waveforms, 1)
            targets = batch['target']

            outputs = model(waveforms)
            loss = F.nll_loss(outputs.squeeze(), targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iteration += 1
            running_loss += loss.item()
            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(targets.data.view_as(pred)).sum()
            total += targets.shape[0]

            pbar.set_postfix(
                {
                    'train loss: ': '{:.4f}'.format(running_loss / iteration),
                    'train accuracy: ': '{:.4f}%'.format(100 * correct / total)
                }
            )

        accuracy = correct / total
        epoch_loss = running_loss / iteration

        print('\nepoch {}'.format(epoch))
        print('train_accuracy: {}\t train_loss: {}\n'.format(accuracy, epoch_loss))
    
    def valid(model, epoch):

        global best_valid_accuracy

        running_loss = 0.0
        iteration = 0
        correct = 0
        total = 0

        model.eval()
        pbar = tqdm(valid_loader, unit="audios", unit_scale=valid_loader.batch_size)

        for batch in pbar:
            
            waveforms = batch['samples']
            waveforms = torch.unsqueeze(waveforms, 1)
            targets = batch['target']

            outputs = model(waveforms)
            loss = F.nll_loss(outputs.squeeze(), targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iteration += 1
            running_loss += loss.item()
            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(targets.data.view_as(pred)).sum()
            total += targets.shape[0]

            pbar.set_postfix(
                {
                    'valid loss: ': '{:.4f}'.format(running_loss / iteration),
                    'valid accuracy: ': '{:.4f}%'.format(100 * correct / total)
                }
            )

        accuracy = correct / total
        epoch_loss = running_loss / iteration

        print('\nepoch {}'.format(epoch))
        print('valid_accuracy: {}\t valid_loss: {}\n'.format(accuracy, epoch_loss))

        save_path = os.path.join(args.save_path, args.first_kernel_size)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if accuracy >= best_valid_accuracy:
            best_valid_accuracy = accuracy
            torch.save(model, os.path.join(save_path, 'vanilla-best-acc.pth'))

        torch.save(model, os.path.join(save_path, 'vanilla-final.pth'))

    for epoch in range(1, args.epochs + 1):
    
        train(model, epoch)
        valid(model, epoch)
        scheduler.step()