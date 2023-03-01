import torch

class AcousticSystem(torch.nn.Module):

    def __init__(self, 
                 classifier: torch.nn.Module, 
                 transform, 
                 defender: torch.nn.Module=None,
                 defense_type: str='wave'):

        super().__init__()

        '''
            the whole audio system: audio -> prediction probability distribution
            
            *defender: audio -> audio or spectrogram -> spectrogram
            *transform: audio -> spectrogram
            *classifier: spectrogram -> prediction probability distribution or 
                            audio -> prediction probability distribution
        '''

        self.classifier = classifier
        self.transform = transform
        self.defender = defender
        self.defense_type = defense_type
        if self.defense_type not in ['wave', 'spec']:
            raise NotImplementedError('argument defense_type should be \'wave\' or \'spec\'!')
    
    def forward(self, x, defend=True):

        # if 0.9 * x.max() > 1 and 0.9 * x.min() < -1:
        #     x = x / (2**15)

        # defense on waveform
        if defend == True and self.defender is not None and self.defense_type == 'wave':
            output = self.defender(x)
        else: 
            output = x
        
        # convert waveform to spectrogram
        if self.transform is not None: 
            output = self.transform(output)

        # defense on spectrogram
        if defend == True and self.defender is not None and self.defense_type == 'spec':
            output = self.defender(output)
        else: 
            output = output
        
        # give prediction of spectrogram
        output = self.classifier(output)

        return output