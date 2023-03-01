import torch

import sys
sys.path.insert(0, './audio_models/ConvNets_SpeechCommands')
sys.path.insert(0, './audio_models/M5')
sys.path.insert(0, './audio_models/RCNN_KWS')

def create_model(path):

    model = torch.load(path)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.float()
    model.eval()

    assert isinstance(model, torch.nn.Module)

    return model