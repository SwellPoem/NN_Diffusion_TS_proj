import os
import sys
import time
import torch
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from Scripts.utility_func import create_instance_from_config, set_seed


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def epoch(dl):
    while True:
        for data in dl:
            yield data

class Trainer(object):
    def __init__(self, model, dataloader, device, train_num_steps, gradient_accumulate_every, save_epoch, results_folder, start_lr, ema_decay, ema_update_every, sc_cfg):
        super().__init__()
        #initialization of the parameters
        self.model = model
        self.device = device
        self.train_num_steps = train_num_steps
        self.gradient_accumulate_every = gradient_accumulate_every
        self.save_epoch = save_epoch
        self.dl = epoch(dataloader['dataloader'])
        self.step = 0
        self.milestone = 0

        #save the checkpoints in the specified folder
        self.results_folder = Path(results_folder + f'_{model.seq_length}')
        os.makedirs(self.results_folder, exist_ok=True)

        # Adam optimizer
        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96])
        # Exponential moving average
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        sc_cfg['params']['optimizer'] = self.opt
        self.sch = create_instance_from_config(sc_cfg)

    #save checkpoint files
    #input: milestone, verbose
    #output: None
    def save(self, milestone, verbose=False):
        #dictionary to contain the current state of the training
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))

    #load checkpoint files
    #input: milestone, verbose
    #output: None
    def load(self, milestone, verbose=False):
        device = self.device
        data = torch.load(str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.milestone = milestone

    # #train the model
    # def train(self, seed):
    #     set_seed(seed)
    #     device = self.device
    #     #training step counter initialization
    #     step = 0

    #     #main loop
    #     with tqdm(initial=step, total=self.train_num_steps) as pbar:
    #         while step < self.train_num_steps:
    #             total_loss = 0.     #initialize the total loss
    #             for _ in range(self.gradient_accumulate_every):
    #                 data = next(self.dl).to(device)     #get the next batch of data
    #                 loss = self.model(data, target=data)        #compute the loss on the current batch
    #                 loss = loss / self.gradient_accumulate_every        #average the loss over the gradient_accumulate_every steps
    #                 loss.backward()     #compute gradients of the loss w.r.t. the model parameters
    #                 total_loss += loss.item()       #update the total loss

    #             pbar.set_description(f'loss: {total_loss:.6f}')

    #             #prevent exploding gradients
    #             clip_grad_norm_(self.model.parameters(), 1.0)
    #             self.opt.step()     #optimization step -> Adam optimizer
    #             self.sch.step(total_loss)       #scheduler step
    #             self.opt.zero_grad()        #reset gradients
    #             self.step += 1      #update global step counter
    #             step += 1       #update local step counter
    #             self.ema.update()

    #             #if the current step is a multiple of the save_epoch, save the current state of the training
    #             with torch.no_grad():
    #                 if self.step != 0 and self.step % self.save_epoch == 0:
    #                     self.milestone += 1
    #                     self.save(self.milestone)

    #             pbar.update(1)

    #     print('training complete')

    #train the model
    def train(self, seed):
        set_seed(seed)
        self._initialize_training()

        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                total_loss = self._compute_loss(pbar)

                self._update_model(total_loss)
                self._check_save_model()

                pbar.update(1)

        print('training complete')

    #training step counter initialization
    def _initialize_training(self):
        self.step = 0

    def _compute_loss(self, pbar):
        total_loss = 0.
        for _ in range(self.gradient_accumulate_every):
            data = next(self.dl).to(self.device)    #get the next batch of data
            loss = self.model(data, target=data)    #compute the loss on the current batch
            loss = loss / self.gradient_accumulate_every    #average the loss over the gradient_accumulate_every steps
            loss.backward()     #compute gradients of the loss w.r.t. the model parameters
            total_loss += loss.item()        #update the total loss

        pbar.set_description(f'loss: {total_loss:.6f}')
        return total_loss

    #prevent exploding gradients
    def _update_model(self, total_loss):
        clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self.sch.step(total_loss)
        self.opt.zero_grad()
        self.step += 1

    #if the current step is a multiple of the save_epoch, save the current state of the training
    def _check_save_model(self):
        with torch.no_grad():
            if self.step != 0 and self.step % self.save_epoch == 0:
                self.milestone += 1
                self.save(self.milestone)

    #sample from the model
    def sample(self, num, size_every, shape=None):
        set_seed(123)
        samples = np.empty([0, shape[0], shape[1]])     #empty array to store the samples
        num_cycle = int(num // size_every) + 1

        #main loop
        for _ in range(num_cycle):
            sample = self.ema.ema_model.generate_mts(batch_size=size_every)     #generate the samples
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()

        return samples

    #restore the model
    #used for forecasting
    def restore(self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        set_seed(1234)
        model_kwargs = {}
        model_kwargs['coef'] = coef
        model_kwargs['learning_rate'] = stepsize
        samples = np.empty([0, shape[0], shape[1]])
        reals = np.empty([0, shape[0], shape[1]])
        masks = np.empty([0, shape[0], shape[1]])

        for idx, (x, t_m) in enumerate(raw_dataloader):
            x, t_m = x.to(self.device), t_m.to(self.device)
            if sampling_steps == self.model.num_timesteps:
                sample = self.ema.ema_model.sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m,
                                                          model_kwargs=model_kwargs)
            else:
                sample = self.ema.ema_model.fast_sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m, model_kwargs=model_kwargs,
                                                               sampling_timesteps=sampling_steps)

            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            reals = np.row_stack([reals, x.detach().cpu().numpy()])
            masks = np.row_stack([masks, t_m.detach().cpu().numpy()])

        return samples, reals, masks