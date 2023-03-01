
import torch.nn as nn
import torch

class EOT(nn.Module):

    def __init__(self, model, loss, EOT_size=1, EOT_batch_size=1, use_grad=True):
        super().__init__()
        self.model = model
        self.loss = loss
        self.EOT_size = EOT_size
        self.EOT_batch_size = EOT_batch_size
        self.EOT_num_batches = self.EOT_size //self.EOT_batch_size
        self.use_grad = use_grad
    
    # def forward(self, x_batch, y_batch, EOT_num_batches=None, EOT_batch_size=None, use_grad=None):
    #     EOT_num_batches = EOT_num_batches if EOT_num_batches else self.EOT_num_batches
    #     EOT_batch_size = EOT_batch_size if EOT_batch_size else self.EOT_batch_size
    def forward(self, x_batch, y_batch, EOT_size=None, EOT_batch_size=None, use_grad=None):
        EOT_size = EOT_size if EOT_size else self.EOT_size
        EOT_batch_size = EOT_batch_size if EOT_batch_size else self.EOT_batch_size
        EOT_num_batches = EOT_size // EOT_batch_size
        use_grad = use_grad if use_grad else self.use_grad
        n_audios, n_channels, max_len = x_batch.size()
        grad = None
        scores = None
        loss = 0
        # decisions = [[]] * n_audios ## wrong, all element shares the same memory
        decisions = [[] for _ in range(n_audios)]
        for EOT_index in range(EOT_num_batches):
            # if EOT_index == EOT_num_batches - 1:
            #     batch_size = EOT_size % EOT_batch_size
            #     if batch_size == 0: break
            # else: 
            batch_size = EOT_batch_size
            x_batch_repeat = x_batch.repeat(batch_size, 1, 1)
            if use_grad:
                x_batch_repeat.retain_grad()
            y_batch_repeat = y_batch.repeat(batch_size)

            scores_EOT = self.model(x_batch_repeat) # scores or logits. Just Name it scores. (batch_size, n_spks)
            decisions_EOT = scores_EOT.max(1, keepdim=True)[1]
            # decisions_EOT, scores_EOT = self.model.make_decision(x_batch_repeat) # scores or logits. Just Name it scores. (batch_size, n_spks)
            loss_EOT = self.loss(scores_EOT, y_batch_repeat)
            if use_grad:
                loss_EOT.backward(torch.ones_like(loss_EOT))

            if EOT_index == 0:
                scores = scores_EOT.view(EOT_batch_size, -1, scores_EOT.shape[1]).mean(0)
                loss = loss_EOT.view(EOT_batch_size, -1).mean(0)
                if use_grad:
                    grad = x_batch_repeat.grad.view(EOT_batch_size, -1, n_channels, max_len).mean(0)
                    x_batch_repeat.grad.zero_()
            else:
                scores.data += scores_EOT.view(EOT_batch_size, -1, scores.shape[1]).mean(0)
                loss.data += loss_EOT.view(EOT_batch_size, -1).mean(0)
                if use_grad:
                    grad.data += x_batch_repeat.grad.view(EOT_batch_size, -1, n_channels, max_len).mean(0)
                    x_batch_repeat.grad.zero_()
            
            decisions_EOT = decisions_EOT.view(EOT_batch_size, -1).detach().cpu().numpy()
            for ii in range(n_audios):
                decisions[ii] += list(decisions_EOT[:, ii])
        
        scores = scores / EOT_num_batches
        loss = loss / EOT_num_batches
        if grad is not None:
            grad = grad / EOT_num_batches
        
        return scores, loss, grad, decisions