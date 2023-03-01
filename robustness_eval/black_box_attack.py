import numpy as np
import torch

from ._EOT import EOT
from ._NES import NES
from ._utils import resolve_loss, resolve_prediction


'''
FAKEBOB attack was proposed in the paper "Who is real Bob? Adversarial Attacks on Speaker Recognition Systems" 
accepted by the conference IEEE S&P (Oakland) 2021.
'''
class FAKEBOB():

    def __init__(self, model, threshold=None,
                task='CSI', targeted=False, confidence=0.5,
                epsilon=0.002, max_iter=200,
                max_lr=0.001, min_lr=1e-6,
                samples_per_draw=50, samples_per_draw_batch_size=50, sigma=0.001, momentum=0.9,
                plateau_length=5, plateau_drop=2.,
                stop_early=True, stop_early_iter=100,
                batch_size=1, EOT_size=1, EOT_batch_size=1, verbose=1):
        
        self.model = model
        self.threshold = threshold
        self.task = task
        self.targeted = targeted
        self.confidence = confidence
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.samples_per_draw = samples_per_draw
        self.samples_per_draw_batch_size = samples_per_draw_batch_size
        self.sigma = sigma
        self.momentum = momentum
        self.plateau_length = plateau_length
        self.plateau_drop = plateau_drop
        self.stop_early = stop_early
        self.stop_early_iter = stop_early_iter
        self.batch_size = batch_size
        self.EOT_size = EOT_size
        self.EOT_batch_size = EOT_batch_size
        self.verbose = verbose

        # loss_name = 'Margin'
        # self.loss, self.grad_sign = resolve_loss(loss_name, targeted, clip_max=False)

    def attack_batch(self, x_batch, y_batch, lower, upper, batch_id):

        with torch.no_grad():
            
            n_audios, _, _ = x_batch.shape

            last_ls = [[]] * n_audios
            lr = [self.max_lr] * n_audios
            prev_loss = [np.infty] * n_audios

            adver_x = x_batch.clone()
            grad = torch.zeros_like(x_batch, dtype=x_batch.dtype, device=x_batch.device)

            best_adver_x = adver_x.clone()
            best_loss = [np.infty] * n_audios
            consider_index = list(range(n_audios))

            success = [False] * n_audios

            for iter in range(self.max_iter + 1):
                prev_grad = grad.clone()
                # loss, grad, adver_loss, scores = self.get_grad(adver_x, y_batch)
                loss, grad, adver_loss, _, y_pred = self.get_grad(adver_x, y_batch)
                # y_pred = torch.max(scores, 1)[1].cpu().numpy()

                # for ii in range(n_audios):
                #     if (y_pred[ii] != y_batch[ii] and self.targeted == False) \
                #         or (y_pred[ii] == y_batch[ii] and self.targeted == True):
                #         success[ii] = True
                for ii, adver_l in enumerate(adver_loss):

                    index = consider_index[ii]

                    if (y_pred[ii] != y_batch[ii] and self.targeted == False) \
                        or (y_pred[ii] == y_batch[ii] and self.targeted == True):
                        success[index] = True
                        
                    if adver_l < best_loss[index]:
                        best_loss[index] = adver_l.cpu().item()
                        best_adver_x[index] = adver_x[ii]

                if self.verbose:
                    print("batch: {} iter: {}, loss: {}, y: {}, y_pred: {}, best loss: {}".format(
                        batch_id, iter, 
                        adver_loss.cpu().numpy(), y_batch.cpu().numpy(), y_pred, best_loss))
                
                # delete alrady found examples
                adver_x, y_batch, prev_grad, grad, lower, upper, \
                consider_index, \
                last_ls, lr, prev_loss, loss = self.delete_found(adver_loss, adver_x, y_batch, prev_grad, grad, lower, upper, 
                                                consider_index, last_ls, lr, prev_loss, loss)
                if adver_x is None: # all found
                    break

                if iter < self.max_iter:
                    grad = self.momentum * prev_grad + (1.0 - self.momentum) * grad
                    for jj, loss_ in enumerate(loss):
                        last_ls[jj].append(loss_)
                        last_ls[jj] = last_ls[jj][-self.plateau_length:]
                        if last_ls[jj][-1] > last_ls[jj][0] and len(last_ls[jj]) == self.plateau_length:
                            if lr[jj] > self.min_lr:
                                lr[jj] = max(lr[jj] / self.plateau_drop, self.min_lr)
                            last_ls[jj] = []
                    
                    lr_t = torch.tensor(lr, device=adver_x.device, dtype=torch.float).unsqueeze(1).unsqueeze(2)
                    adver_x.data = adver_x + self.grad_sign * lr_t * torch.sign(grad)
                    adver_x.data = torch.min(torch.max(adver_x.data, lower), upper)

                    if self.stop_early and iter % self.stop_early_iter == 0:
                        loss_np = np.array([l.cpu() for l in loss])
                        converge_loss = np.array(prev_loss) * 0.9999 - loss_np
                        adver_x, y_batch, prev_grad, grad, lower, upper, \
                        consider_index, \
                        last_ls, lr, prev_loss, loss = self.delete_found(converge_loss, adver_x, y_batch, prev_grad, grad, lower, upper, 
                                                consider_index, last_ls, lr, prev_loss, loss)
                        if adver_x is None: # all converage
                            break

                        prev_loss = loss_np          
            
            # for kk, best_l in enumerate(best_loss):
            #     if best_l < 0:
            #         success[kk] = True
            
            return best_adver_x, success
    
    def delete_found(self, adver_loss, adver_x, y_batch, prev_grad, grad, lower, upper, 
                    consider_index, last_ls, lr, prev_loss, loss):
        adver_x_u = None
        y_batch_u = None
        prev_grad_u = None
        grad_u = None
        lower_u = None
        upper_u = None

        consider_index_u = []
        last_ls_u = []
        lr_u = []
        prev_loss_u = []
        loss_u = []
        
        for ii, adver_l in enumerate(adver_loss):
            if adver_l < 0:
                pass
            else:
                if adver_x_u is None:
                    adver_x_u = adver_x[ii:ii+1, ...]
                    y_batch_u = y_batch[ii:ii+1]
                    prev_grad_u = prev_grad[ii:ii+1, ...]
                    grad_u = grad[ii:ii+1, ...]
                    lower_u = lower[ii:ii+1, ...]
                    upper_u = upper[ii:ii+1, ...]
                else:
                    adver_x_u = torch.cat((adver_x_u, adver_x[ii:ii+1, ...]), 0)
                    y_batch_u = torch.cat((y_batch_u, y_batch[ii:ii+1]))
                    prev_grad_u = torch.cat((prev_grad_u, prev_grad[ii:ii+1, ...]), 0)
                    grad_u = torch.cat((grad_u, grad[ii:ii+1, ...]), 0)
                    lower_u = torch.cat((lower_u, lower[ii:ii+1, ...]), 0)
                    upper_u = torch.cat((upper_u, upper[ii:ii+1, ...]), 0)
                index = consider_index[ii]
                consider_index_u.append(index)
                last_ls_u.append(last_ls[ii])
                lr_u.append(lr[ii])
                prev_loss_u.append(prev_loss[ii])
                loss_u.append(loss[ii])

        return adver_x_u, y_batch_u, prev_grad_u, \
                grad_u, lower_u, upper_u, \
                consider_index_u, \
                last_ls_u, lr_u, prev_loss_u, loss_u

    def get_grad(self, x, y):
        NES_wrapper = NES(self.samples_per_draw, self.samples_per_draw_batch_size, self.sigma, self.EOT_wrapper)
        mean_loss, grad, adver_loss, adver_score, predict = NES_wrapper(x, y)
        
        return mean_loss, grad, adver_loss, adver_score, predict

    def generate(self, x, y, targeted=False):

        if self.task in ['SV', 'OSI'] and self.threshold is None:
            raise NotImplementedError('You are running black box attack for {} task, \
                        but the threshold not specified. Consider calling estimate threshold')
        self.targeted = targeted
        self.loss, self.grad_sign = resolve_loss('Margin', self.targeted, self.confidence, self.task, self.threshold, False)
        self.EOT_wrapper = EOT(self.model, self.loss, self.EOT_size, self.EOT_batch_size, False)

        lower = -1
        upper = 1
        assert lower <= x.max() < upper, 'generating adversarial examples should be done in [-1, 1) float domain' 
        n_audios, n_channels, _ = x.size()
        assert n_channels == 1, 'Only Support Mono Audio'
        assert y.shape[0] == n_audios, 'The number of x and y should be equal' 
        upper = torch.clamp(x+self.epsilon, max=upper)
        lower = torch.clamp(x-self.epsilon, min=lower)

        batch_size = min(self.batch_size, n_audios)
        n_batches = int(np.ceil(n_audios / float(batch_size)))
        for batch_id in range(n_batches):
            x_batch = x[batch_id*batch_size:(batch_id+1)*batch_size] # (batch_size, 1, max_len)
            y_batch = y[batch_id*batch_size:(batch_id+1)*batch_size]
            lower_batch = lower[batch_id*batch_size:(batch_id+1)*batch_size]
            upper_batch = upper[batch_id*batch_size:(batch_id+1)*batch_size]
            adver_x_batch, success_batch = self.attack_batch(x_batch, y_batch, lower_batch, upper_batch, batch_id)
            if batch_id == 0:
                adver_x = adver_x_batch
                success = success_batch
            else:
                adver_x = torch.cat((adver_x, adver_x_batch), 0)
                success += success_batch

        return adver_x, success
    
    def estimate_threshold_run(self, x, step=0.1):

        n_audios, _, _ = x.shape

        # d, s = self.model.make_decision(x)
        s = self.model(x)
        d = s.max(1, keepdim=True)[1]
        d = d[0]
        s = s[0]
        if d != -1:
            return # aleady accept, cannot be used to estimate threshold
        y = torch.tensor([-1] * n_audios, dtype=torch.long, device=x.device)
        init_score = np.max(s.cpu().numpy())
        delta = np.abs(init_score * step)
        threshold = init_score + delta

        adver_x = x.clone()
        grad = torch.zeros_like(x, dtype=x.dtype, device=x.device)

        lower = -1
        upper = 1
        upper = torch.clamp(x+self.epsilon, max=upper)
        lower = torch.clamp(x-self.epsilon, min=lower)

        iter_outer = 0
        n_iters = 0

        while True:
            self.loss, self.grad_sign = resolve_loss('Margin', False, 0., self.task, threshold, False)
            self.EOT_wrapper = EOT(self.model, self.loss, self.EOT_size, self.EOT_batch_size, False)

            iter_inner = 0

            last_ls = [[]] * n_audios
            lr = [self.max_lr] * n_audios

            while True:

                # test whether succeed
                # decision, score = self.model.make_decision(adver_x)
                score = self.model(adver_x)
                decision = score.max(1, keepdim=True)[1]

                decision = decision[0]
                score = score[0]
                score = np.max(score.cpu().numpy())
                print(iter_outer, iter_inner, score, self.model.threshold)
                if decision != -1: # succeed, found the threshold
                    return score
                elif score >= threshold: # exceed the candidate threshold, but not succeed, exit the inner loop and increase the threshold
                    break

                # not succeed, update
                prev_grad = grad.clone()
                loss, grad, _, _, _ = self.get_grad(adver_x, y)

                grad = self.momentum * prev_grad + (1.0 - self.momentum) * grad
                for jj, loss_ in enumerate(loss):
                    last_ls[jj].append(loss_)
                    last_ls[jj] = last_ls[jj][-self.plateau_length:]
                    if last_ls[jj][-1] > last_ls[jj][0] and len(last_ls[jj]) == self.plateau_length:
                        if lr[jj] > self.min_lr:
                            lr[jj] = max(lr[jj] / self.plateau_drop, self.min_lr)
                        last_ls[jj] = []
                
                lr_t = torch.tensor(lr, device=adver_x.device, dtype=torch.float).unsqueeze(1).unsqueeze(2)
                adver_x.data = adver_x + self.grad_sign * lr_t * torch.sign(grad)
                adver_x.data = torch.min(torch.max(adver_x.data, lower), upper)

                iter_inner += 1
                n_iters += 1
            
            threshold += delta
            iter_outer += 1

    def estimate_threshold(self, x, step=0.1):
        if self.task == 'CSI':
            print("--- Warning: no need to estimate threshold for CSI, quitting ---")
            return
        
        with torch.no_grad():
            estimated_thresholds = []
            for xx in x.unsqueeze(0): # parallel running, not easy for batch running
                estimated_threshold = self.estimate_threshold_run(xx, step)
                if estimated_threshold is not None:
                    estimated_thresholds.append(estimated_threshold)
            if len(estimated_thresholds) > 0:
                self.threshold = np.mean(estimated_thresholds)
            else:
                self.threshold = None
            return self.threshold

# not used
# region
# class SirenAttack():

#     def __init__(self, model, threshold=None, 
#                 task='CSI', targeted=False, confidence=0.,
#                 epsilon=0.002, max_epoch=300, max_iter=30,
#                 c1=1.4961, c2=1.4961, n_particles=25, w_init=0.9, w_end=0.1,
#                 batch_size=1, EOT_size=1, EOT_batch_size=1, verbose=1, abort_early=True, abort_early_iter=10, abort_early_epoch=10):
        
#         self.model = model
#         self.threshold = threshold
#         self.task = task
#         self.targeted = targeted
#         self.confidence = confidence
#         self.epsilon = epsilon
#         self.max_epoch = max_epoch
#         self.max_iter = max_iter
#         self.c1 = c1
#         self.c2 = c2
#         self.n_particles = n_particles
#         self.w_init = w_init
#         self.w_end = w_end
#         self.batch_size = batch_size
#         self.EOT_size = EOT_size
#         self.EOT_batch_size = EOT_batch_size
#         self.verbose = verbose

#         self.abort_early = abort_early
#         self.abort_early_iter = abort_early_iter
#         self.abort_early_epoch = abort_early_epoch
    

#     def attack_batch(self, x_batch, y_batch, lower, upper, batch_id):

#         with torch.no_grad():

#             v_upper = torch.abs(lower - upper)
#             v_lower = -v_upper

#             x_batch_clone = x_batch.clone() # for return
#             n_audios, n_channels, N = x_batch.shape
#             consider_index = list(range(n_audios))
#             # pbest_locations = np.random.uniform(low=lower.unsqueeze(1).cpu().numpy(),
#             #                     high=upper.unsqueeze(1).cpu().numpy(), size=(n_audios, self.n_particles, n_channels, N))
#             # pbest_locations = torch.tensor(pbest_locations, device=x_batch.device, dtype=torch.float)

#             gbest_location = torch.zeros(n_audios, n_channels, N, dtype=torch.float, device=x_batch.device)
#             gbests = torch.ones(n_audios, device=x_batch.device, dtype=torch.float) * np.infty
#             gbest_predict = np.array([None] * n_audios)
#             prev_gbest = gbests.clone()
#             prev_gbest_epoch = gbests.clone()

#             success = [False] * n_audios
#             continue_flag = True
#             for epoch in range(self.max_epoch):
                
#                 if not continue_flag:
#                     break

#                 if epoch == 0:
#                     pbest_locations = np.random.uniform(low=lower.unsqueeze(1).cpu().numpy(),
#                                 high=upper.unsqueeze(1).cpu().numpy(), size=(n_audios, self.n_particles, n_channels, N))
#                     pbest_locations = torch.tensor(pbest_locations, device=x_batch.device, dtype=torch.float)
#                     pbests = torch.ones(n_audios, self.n_particles, device=x_batch.device, dtype=torch.float) * np.infty
#                 else:
#                     best_index = torch.argmin(pbests, dim=1) # (len(consider_index), )
#                     best_location = pbest_locations[np.arange(len(consider_index)), best_index] # (len(consider_index), n_channels, N)
#                     pbest_locations = np.random.uniform(low=lower.unsqueeze(1).cpu().numpy(),
#                                 high=upper.unsqueeze(1).cpu().numpy(), size=(len(consider_index), self.n_particles-1, n_channels, N))
#                     pbest_locations = torch.tensor(pbest_locations, device=x_batch.device, dtype=torch.float)
#                     pbest_locations = torch.cat((best_location.unsqueeze(1), pbest_locations), dim=1)
#                     pbests_new = torch.ones(len(consider_index), self.n_particles-1, device=x_batch.device, dtype=torch.float) * np.infty
#                     pbests = torch.cat((pbests[np.arange(len(consider_index)), best_index].unsqueeze(1), pbests_new), dim=1)

#                 locations = pbest_locations.clone()
#                 # volicities = np.random.uniform(low=lower.unsqueeze(1).cpu().numpy(),
#                 #                 high=upper.unsqueeze(1).cpu().numpy(), size=(len(consider_index), self.n_particles, n_channels, N))
#                 volicities = np.random.uniform(low=v_lower.unsqueeze(1).cpu().numpy(),
#                                 high=v_upper.unsqueeze(1).cpu().numpy(), size=(len(consider_index), self.n_particles, n_channels, N))
#                 volicities = torch.tensor(volicities, device=x_batch.device, dtype=torch.float)

#                 ### ????
#                 # pbests = torch.ones(len(consider_index), self.n_particles, device=x_batch.device, dtype=torch.float) * np.infty

#                 continue_flag_inner = True

#                 # for iter in range(self.max_iter):
#                 for iter in range(self.max_iter+1):

#                     if not continue_flag_inner:
#                         break

#                     eval_x = locations + x_batch.unsqueeze(1) # (n_audios, self.n_particles, n_channels, N)
#                     eval_x = eval_x.view(-1, n_channels, N)
#                     eval_y = None
#                     for jj, y_ in enumerate(y_batch):
#                         tmp = torch.tensor([y_] * self.n_particles, dtype=torch.long, device=x_batch.device)
#                         if jj == 0:
#                             eval_y = tmp
#                         else:
#                             eval_y = torch.cat((eval_y, tmp))
#                     # print(eval_x.shape, eval_y.shape)
#                     _, loss, _, decisions = self.EOT_wrapper(eval_x, eval_y)
#                     EOT_num_batches = int(self.EOT_wrapper.EOT_size // self.EOT_wrapper.EOT_batch_size)
#                     loss.data /= EOT_num_batches # (n_audios*n_p,)
#                     loss = loss.view(len(consider_index), -1) # (n_audios, n_p)
#                     predict = resolve_prediction(decisions).reshape(len(consider_index), -1)

#                     update_index = torch.where(loss < pbests)
#                     update_ii = update_index[0].cpu().numpy().tolist()
#                     update_jj = update_index[1].cpu().numpy().tolist()
#                     if len(update_ii) > 0:
#                         for ii, jj in zip(update_ii, update_jj):
#                             pbests[ii, jj] = loss[ii, jj]
#                             pbest_locations[ii, jj, ...] = locations[ii, jj, ...]
                    
#                     # if self.abort_early and (iter+1) % self.abort_early_iter == 0:
#                     #     prev_gbest.data = gbests
                    
#                     gbest_index = torch.argmin(pbests, 1)
#                     for kk in range(gbest_index.shape[0]):
#                         index = consider_index[kk]
#                         if pbests[kk, gbest_index[kk]] < gbests[index]:
#                             gbests[index] = pbests[kk, gbest_index[kk]]
#                             gbest_location[index] = pbest_locations[kk, gbest_index[kk]]
#                             gbest_predict[index] = predict[kk, gbest_index[kk]]
                    
#                     # for pp in range(len(y_batch)):
#                     #     if (gbest_predict[consider_index][pp] != y_batch[pp] and self.targeted == False) \
#                     #         or (gbest_predict[consider_index][pp] == y_batch[pp] and self.targeted == True):
#                     #         success[pp] = True

#                     if self.verbose:
#                         print('batch: {}, epoch: {}, iter: {}, y: {}, y_pred: {}, gbest: {}'.format(batch_id,
#                             epoch, iter, y_batch.cpu().numpy().tolist(), gbest_predict[consider_index], gbests[consider_index].cpu().numpy().tolist()))
                    
#                     if self.abort_early and (iter+1) % self.abort_early_iter == 0:
#                         if torch.mean(gbests) > 0.9999 * torch.mean(prev_gbest):
#                             print('Converge, Break Inner Loop')
#                             continue_flag_inner = False
#                             # break
#                         # prev_gbest.data = gbests
#                         prev_gbest = gbests.clone()

#                     # stop early
#                     # x_batch, y_batch, lower, upper
#                     # pbest_locations, locations, v, pbests
#                     # consider_index
#                     # delete alrady found examples
#                     x_batch, y_batch, lower, upper, \
#                     pbest_locations, locations, volicities, pbests, \
#                     consider_index = self.delete_found(gbests[consider_index], x_batch, y_batch, lower, upper, 
#                                                     pbest_locations, locations, volicities, pbests, 
#                                                     consider_index)
#                     if len(consider_index) == 0:
#                         continue_flag = False # used to break the outer loop
#                         break
#                     else:
#                         v_upper = torch.abs(lower - upper)
#                         v_lower = -v_upper

#                     if iter < self.max_iter:
#                         w = (self.w_init - self.w_end) * (self.max_iter - iter - 1) / self.max_iter + self.w_end
#                         # r1 = np.random.rand() + 0.00001
#                         # r2 = np.random.rand() + 0.00001
#                         r1 = np.random.rand(len(consider_index), self.n_particles, n_channels, N) + 0.00001
#                         r2 = np.random.rand(len(consider_index), self.n_particles, n_channels, N) + 0.00001
#                         r1 = torch.tensor(r1, device=x_batch.device, dtype=torch.float)
#                         r2 = torch.tensor(r2, device=x_batch.device, dtype=torch.float)
#                         volicities = (w * volicities + self.c1 * r1 * (pbest_locations - locations) +
#                                 self.c2 * r2 * (gbest_location[consider_index, ...].unsqueeze(1) - locations))
#                         locations = locations + volicities
#                         locations = torch.min(torch.max(locations, lower.unsqueeze(1)), upper.unsqueeze(1))
                
#                 if self.abort_early and (epoch+1) % self.abort_early_epoch == 0:
#                     if torch.mean(gbests) > 0.9999 * torch.mean(prev_gbest_epoch):
#                         print('Converge, Break Outer Loop')
#                         continue_flag = False
#                         # break
#                     prev_gbest_epoch = gbests.clone()
            
#             success = [False] * n_audios
#             for kk, best_l in enumerate(gbests):
#                 if best_l < 0:
#                     success[kk] = True

#             return gbest_location + x_batch_clone, success

        
#     def delete_found(self, gbests, x_batch, y_batch, lower, upper, 
#                     pbest_locations, locations, volicities, pbests, 
#                     consider_index):
        
#         x_batch_u = None
#         y_batch_u = None
#         lower_u = None
#         upper_u = None 
#         pbest_locations_u = None
#         locations_u = None
#         volicities_u = None
#         pbests_u = None
#         consider_index_u = []

#         for ii, g in enumerate(gbests):
#             if g < 0:
#                 continue
#             else:
#                 if x_batch_u is None:
#                     x_batch_u = x_batch[ii:ii+1]
#                     y_batch_u = y_batch[ii:ii+1]
#                     lower_u = lower[ii:ii+1]
#                     upper_u = upper[ii:ii+1]
#                     pbest_locations_u = pbest_locations[ii:ii+1]
#                     locations_u = locations[ii:ii+1]
#                     volicities_u = volicities[ii:ii+1]
#                     pbests_u = pbests[ii:ii+1]
#                 else:
#                     x_batch_u = torch.cat((x_batch_u, x_batch[ii:ii+1]), 0)
#                     y_batch_u = torch.cat((y_batch_u, y_batch[ii:ii+1]))
#                     lower_u = torch.cat((lower_u, lower[ii:ii+1]), 0)
#                     upper_u = torch.cat((upper_u, upper[ii:ii+1]), 0)
#                     pbest_locations_u = torch.cat((pbest_locations_u, pbest_locations[ii:ii+1]), 0)
#                     locations_u = torch.cat((locations_u, locations[ii:ii+1]), 0)
#                     volicities_u = torch.cat((volicities_u, volicities[ii:ii+1]), 0)
#                     pbests_u = torch.cat((pbests_u, pbests[ii:ii+1]), 0)
#                 index = consider_index[ii]
#                 consider_index_u.append(index)
        
#         return x_batch_u, y_batch_u, lower_u, upper_u, \
#                 pbest_locations_u, locations_u, volicities_u, pbests_u, \
#                 consider_index_u
    

#     def generate(self, x, y, targeted=False):

#         if self.task in ['SV', 'OSI'] and self.threshold is None:
#             raise NotImplementedError('You are running black box attack for {} task, \
#                         but the threshold not specified. Consider Estimating the threshold by FAKEBOB!')
#         self.targeted = targeted
#         self.loss, self.grad_sign = resolve_loss('Margin', self.targeted, self.confidence, self.task, self.threshold, False)
#         self.EOT_wrapper = EOT(self.model, self.loss, self.EOT_size, self.EOT_batch_size, False)

#         lower = -1
#         upper = 1
#         assert lower <= x.max() < upper, 'generating adversarial examples should be done in [-1, 1) float domain' 
#         n_audios, n_channels, _ = x.size()
#         assert n_channels == 1, 'Only Support Mono Audio'
#         assert y.shape[0] == n_audios, 'The number of x and y should be equal' 
#         # upper = torch.clamp(x+self.epsilon, max=upper)
#         # lower = torch.clamp(x-self.epsilon, min=lower)
#         lower = torch.clamp(-1-x, min=-self.epsilon) # for distortion, not adver audio
#         upper = torch.clamp(1-x, max=self.epsilon) # for distortion, not adver audio

#         batch_size = min(self.batch_size, n_audios)
#         n_batches = int(np.ceil(n_audios / float(batch_size)))
#         for batch_id in range(n_batches):
#             x_batch = x[batch_id*batch_size:(batch_id+1)*batch_size] # (batch_size, 1, max_len)
#             y_batch = y[batch_id*batch_size:(batch_id+1)*batch_size]
#             lower_batch = lower[batch_id*batch_size:(batch_id+1)*batch_size]
#             upper_batch = upper[batch_id*batch_size:(batch_id+1)*batch_size]
#             adver_x_batch, success_batch = self.attack_batch(x_batch, y_batch, lower_batch, upper_batch, batch_id)
#             if batch_id == 0:
#                 adver_x = adver_x_batch
#                 success = success_batch
#             else:
#                 adver_x = torch.cat((adver_x, adver_x_batch), 0)
#                 success += success_batch

#         return adver_x, success

# from ._Kenan import atk_bst
# from ._KenanFFT import atk_bst_fft
# class Kenansville(object):

#     def __init__(self, model, atk_name='fft', max_iter=15, raster_width=100, early_stop=False, targeted=False, verbose=1, BITS=16, batch_size=1):

#         self.model = model # remember to call model.eval()
#         self.atk_name = atk_name
#         self.max_iter = max_iter
#         self.raster_width = raster_width #  for ssa
#         self.targeted = targeted
#         self.verbose = verbose
#         self.BITS = BITS
#         self.early_stop = early_stop

#         if atk_name == 'ssa': # ssa attack does not support batch attack
#             self.batch_size =1
#         else:
#             self.batch_size = batch_size

#     def attack_batch(self, x_batch, y_batch, batch_id, fs=16_000):

#         if self.atk_name == 'ssa':
#             device = x_batch.device
#             shape = x_batch.shape
#             x_batch = x_batch.cpu().numpy()
#             if 0.9 * x_batch.max() <= 1 and 0.9 * x_batch.min() >= -1:
#                 x_batch = x_batch * (2 ** (self.BITS-1))
#             x_batch = x_batch.astype(np.int16).flatten()

#             x_adv, success = atk_bst(x_batch, fs, y_batch.item(), self.targeted, 
#                                 [self.raster_width], [self.model], [self.atk_name], self.max_iter, self.verbose, self.early_stop)

#             # return x_batch, success
#             # return torch.from_numpy(x_adv/2**(BITS-1)).to(device).view(shape), [success]
#             return x_adv.reshape(1, 1, -1), [success]
#         elif self.atk_name == 'fft':
#             device = x_batch.device
#             shape = x_batch.shape
#             # if 0.9 * x_batch.max() <= 1 and 0.9 * x_batch.min() >= -1:
#             #     x_batch = x_batch * (2 ** (self.BITS-1))
#             #     scale = True
#             # else:
#             #     scale = False
#             x_adv, success = atk_bst_fft(x_batch, fs, y_batch, self.targeted, 
#                                 [self.raster_width], [self.model], [self.atk_name], self.max_iter, self.verbose, self.early_stop)
#             # if scale:
#             #     x_adv = x_adv / (2 ** (self.BITS-1))
#             # # print(x_batch.shape, x_adv.shape)
#             return x_adv, success
            

#     def generate(self, x, y, targeted=False, fs=16_000):

#         self.targeted = targeted

#         n_audios, n_channels, _ = x.size()
#         assert n_channels == 1, 'Only Support Mono Audio'
#         assert y.shape[0] == n_audios, 'The number of x and y should be equal' 

#         # self.batch_size = 1 # Kenan not supports batch attack
#         batch_size = min(self.batch_size, n_audios)
#         n_batches = int(np.ceil(n_audios / float(batch_size)))
#         for batch_id in range(n_batches):
#             x_batch = x[batch_id*batch_size:(batch_id+1)*batch_size] # (batch_size, 1, max_len)
#             y_batch = y[batch_id*batch_size:(batch_id+1)*batch_size]
#             adver_x_batch, success_batch = self.attack_batch(x_batch, y_batch, batch_id, fs=fs)
#             # print(adver_x_batch.shape)
#             if batch_id == 0:
#                 adver_x = adver_x_batch
#                 success = success_batch
#             else:
#                 if type(adver_x) == torch.Tensor:
#                     adver_x = torch.cat((adver_x, adver_x_batch), 0)
#                 elif type(adver_x) == np.ndarray:
#                     adver_x = np.concatenate((adver_x, adver_x_batch), 0)

#                 success += success_batch
#         # print(adver_x.shape)
#         return adver_x, success

# endregion