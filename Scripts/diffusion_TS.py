import math
import torch
import torch.nn.functional as F

from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from functools import partial
from Scripts.transformer_model import Transformer
from Scripts.utility_func import default, identity, extract

# helper function to generate beta schedules
# taken from original code
# cosine beta schedule
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule for beta values
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class Diffusion_TS(nn.Module):
    # parameters: length of the sequence, size of feature vector, number of encoder layers, number of decoder layers, dimension of the model, number of timesteps, number of sampling timesteps, loss function type, schedule for beta parameter, number of attention heads, multiplier for the hidden layer size, 
    # eta, dropout rate for attention mechanism, dropout rate for residual connections, kernel size of the convolutional layers, padding size, use feed forward, regularization weight
    def __init__(self, seq_length, feature_size, n_layer_enc=3, n_layer_dec=6, d_model=None, timesteps=1000, sampling_timesteps=None, loss_type='l1', beta_schedule='cosine', n_heads=4, mlp_hidden_times=4, eta=0., attn_pd=0., resid_pd=0., kernel_size=None, padding_size=None, use_ff=True, reg_weight=None):
        super(Diffusion_TS, self).__init__()

        #initialize model parameters
        self.eta, self.use_ff = eta, use_ff
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.ff_weight = default(reg_weight, math.sqrt(self.seq_length) / 5)

        #initialize transformer model
        self.model = Transformer(n_feat=feature_size, n_channel=seq_length, n_layer_enc=n_layer_enc, n_layer_dec=n_layer_dec,
                                 n_heads=n_heads, attn_drop_prob=attn_pd, resid_drop_prob=resid_pd, mlp_hidden_times=mlp_hidden_times,
                                 max_len=seq_length, embd_dim=d_model, conv_params=[kernel_size, padding_size])

        #initialize beta schedule for the diffusion process
        if beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        
        #calculate alpha values from beta values and their cumulative products
        alphas = 1. - betas
        alphas_cumulative = torch.cumprod(alphas, dim=0)    #cumulative product of the alpha up to t
        alphas_cumulative_prev = F.pad(alphas_cumulative[:-1], (1, 0), value=1.)    #cumulative product of the alpha up to t-1

        #see eqn 18
        posterior_variance = betas * (1. - alphas_cumulative_prev) / (1. - alphas_cumulative)   # equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        #number of timesteps in the diffusion process
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)     #timesteps stored as instance variable
        self.loss_type = loss_type         #loss type stored as instance variable

        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps)  # default num sampling timesteps to number of timesteps at training

        # check if number of sampling timesteps is less than or equal to number of timesteps
        assert self.sampling_timesteps <= timesteps
        # check if fast sampling is possible => number of sampling timesteps is less than total timesteps
        self.fast_sampling = self.sampling_timesteps < timesteps

        # helper function to register a tensor as a buffer in the model and convert it to torch.float32 data type
        #register_buffer is a library function in pytorch
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        #use the helper function to register the tensors as buffers
        register_buffer('betas', betas)
        register_buffer('alphas_cumulative', alphas_cumulative)
        register_buffer('alphas_cumulative_prev', alphas_cumulative_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others and register them as buffers
        register_buffer('sqrt_alphas_cumulative', torch.sqrt(alphas_cumulative))
        register_buffer('sqrt_one_minus_alphas_cumulative', torch.sqrt(1. - alphas_cumulative))
        register_buffer('log_one_minus_alphas_cumulative', torch.log(1. - alphas_cumulative))
        register_buffer('sqrt_recip_alphas_cumulative', torch.sqrt(1. / alphas_cumulative))
        register_buffer('sqrt_recipm1_alphas_cumulative', torch.sqrt(1. / alphas_cumulative - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0) and others and register them as buffers
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))   #log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumulative_prev) / (1. - alphas_cumulative))   #first coefficient of the posterior mean -> see eqn 18
        register_buffer('posterior_mean_coef2', (1. - alphas_cumulative_prev) * torch.sqrt(alphas) / (1. - alphas_cumulative))   #second coefficient of the posterior mean -> see eqn 18

        # calculate reweighting factors for the loss function and register them as buffers
        register_buffer('loss_weight', torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumulative) / betas / 100)

    # calculate noise at given timestep t based on the initial state and the current state
    # noise = difference between the current state and the state
    # scaled by the square root of the reciprocal of the cumulative alpha values
    # divided by the square root of the reciprocal of the cumulative alpha values minus 1
    # EQUATION 8 IN THE PAPER
    #input: x_t -> current state, t -> current timestep, x0 -> initial state
    #output: noise at timestep t
    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumulative, t, x_t.shape) * x_t - x0) / extract(self.sqrt_recipm1_alphas_cumulative, t, x_t.shape)
        )
    
    #predict the start state based on the current state, the current timestep and the noise
    # start state = difference between the current state and the noise scaled
    #reversed process from eq 15 to get x0 starting from the noise x_T that is the current noisy state
    #EQUATION 15 IN THE PAPER
    #input: x_t -> current state, t -> current timestep, noise -> noise at timestep t
    #output: start state at timestep t
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumulative, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumulative, t, x_t.shape) * noise
        )

    #calculate mean and variance of the posterior distribution q(x_{t-1} | x_t, x_0)
    #EQUATION 18 IN THE PAPER
    #input: x_start -> initial state, x_t -> current state, t -> current timestep
    #output: mean, variance and log variance of the posterior distribution
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    #calculate the output of the model at a given timestep t
    #EQUATION 7 IN THE PAPER
    #input: x -> current state, t -> current timestep
    #output: model output at timestep t
    def output(self, x, t, padding_masks=None):
        trend, season = self.model(x, t, padding_masks=padding_masks)
        model_output = trend + season
        return model_output

    #calculate the predicted noise and the initial state at a given timestep t based on the current state
    #input: x -> current state, t -> current timestep
    #output: predicted noise and initial state at timestep t
    def model_predictions(self, x, t, clip_x_start=False, padding_masks=None):
        if padding_masks is None:
            padding_masks = torch.ones(x.shape[0], self.seq_length, dtype=bool, device=x.device)

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        x_start = self.output(x, t, padding_masks)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    #calculate the initial state based on the current state using model_predictions function 
    # then calls q_posterior to calculate the parameters of the posterior distribution
    #input: x -> current state, t -> current timestep
    #output: mean, variance and log variance of the posterior distribution and the initial state at timestep t
    def p_mean_variance(self, x, t, clip_denoised=True):
        _, x_start = self.model_predictions(x, t)
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    #generate a sample from the posterior distriution at a given timestep t based on the current state
    #input: x -> current state, t -> current timestep
    #output: sample from the posterior distribution at timestep t
    def p_sample(self, x, t: int, clip_denoised=True):
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    #iteratively apply the p_sample function to generate a sample from the posterior distribution at each timestep
    #the p_sample is applied in the reversed time order
    #input: shape -> shape of the sample to be generated
    #output: sample generated from the posterior distribution at each timestep
    @torch.no_grad()    #to reduce memory usage and speed up computation by not storing the gradients and not updating the model parameters since it is only used for inference
    def sample(self, shape):
        device = self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            img, _ = self.p_sample(img, t)
        return img

    # generate a sample by starting with a tensort of random noise and iteratively applying the model to generate the sample as before
    # difference: uses a subset of the timesteps to generate the sample, faster that the preious method
    #input: shape -> shape of the sample to be generated
    #output: sample generated from the posterior distribution at each timestep
    @torch.no_grad()
    def fast_sample(self, shape, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumulative[time]
            alpha_next = self.alphas_cumulative[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return img

    #generate multiple time series samples from a model
    #can either use sample or fast_sample to generate the samples
    #input: batch_size -> number of samples to generate
    #output: multiple time series samples
    def generate_mts(self, batch_size=16):
        feature_size, seq_length = self.feature_size, self.seq_length
        sample_fn = self.fast_sample if self.fast_sampling else self.sample
        return sample_fn((batch_size, seq_length, feature_size))

    #dynamically determinate the loss function based on the loss type declared
    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss    # takes the mean element-wise absolute value difference
        elif self.loss_type == 'l2':
            return F.mse_loss   # measures the element-wise mean squared error
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    #generate a sample from the transition distribution q(x_t | x_{t-1}) at a given timestep t based on the current state
    #input: x_start -> initial state, t -> current timestep, noise -> noise at timestep t
    #output: sample from the transition distribution at timestep t
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return ( extract(self.sqrt_alphas_cumulative, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumulative, t, x_start.shape) * noise )

    #calculate training loss of the model based on current state, target and some noise
    def _train_loss(self, x_start, t, target=None, noise=None, padding_masks=None):
        #default noise is a tensor of random values with the same shape as x_start
        noise = default(noise, lambda: torch.randn_like(x_start))
        if target is None:
            target = x_start

        #generate a sample from the transition distribution using the q_sample function
        x = self.q_sample(x_start=x_start, t=t, noise=noise)  # noise sample
        model_out = self.output(x, t, padding_masks)    #get the model's prediction

        #calculate the loss based on the model's prediction and the target
        train_loss = self.loss_fn(model_out, target, reduction='none')

        #calculate the fourier loss if the model uses the fourier features
        #fourier loss = difference between fourier transforms of the model's prediction and the target
        #input: model_out -> model's prediction, target -> target
        #output: fourier loss
        fourier_loss = torch.tensor([0.])
        if self.use_ff:
            fft1 = torch.fft.fft(model_out.transpose(1, 2), norm='forward')
            fft2 = torch.fft.fft(target.transpose(1, 2), norm='forward')
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none') + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none')
            train_loss +=  self.ff_weight * fourier_loss
        
        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        return train_loss.mean()

    #computes the forward pass of the model by generating a tensor of random timesteps
    # and calculating the training loss based on the input and these timesteps
    #input: x -> current state
    #output: training loss
    def forward(self, x, **kwargs):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self._train_loss(x_start=x, t=t, **kwargs)

    #generate a sample and decompose it into trend, season and residual components
    #input: x -> current state, t -> current timestep
    #output: trend, season, residual and the sample
    def return_components(self, x, t: int):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.tensor([t])
        t = t.repeat(b).to(device)
        x = self.q_sample(x, t)
        trend, season, residual = self.model(x, t, return_res=True)
        return trend, season, residual, x

    #for imputation task
    #generate a sample and impute the missing values in the sample
    #input: x -> current state, t -> current timestep, partial_mask -> mask for missing values
    #output: imputed sample
    def fast_sample_infill(self, shape, target, sampling_timesteps, partial_mask=None, clip_denoised=True, model_kwargs=None):
        batch, device, total_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='conditional sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumulative[time]
            alpha_next = self.alphas_cumulative[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(img)

            img = pred_mean + sigma * noise
            img = self.langevin_func(sample=img, mean=pred_mean, sigma=sigma, t=time_cond,
                                   tgt_embs=target, partial_mask=partial_mask, **model_kwargs)
            target_t = self.q_sample(target, t=time_cond)
            img[partial_mask] = target_t[partial_mask]

        img[partial_mask] = target[partial_mask]

        return img

    #for imputation or forecasting tasks
    #generates a sample from the model by iteratively updating an image based on the model's predictions and a tensor of random noise
    #input: shape -> shape of the sample to be generated, target -> target, partial_mask -> mask for missing values
    #output: sample generated from the model
    def sample_infill(self, shape, target, partial_mask=None, clip_denoised=True, model_kwargs=None):
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='conditional sampling loop time step', total=self.num_timesteps):
            img = self.p_sample_infill(x=img, t=t, clip_denoised=clip_denoised, target=target,
                                       partial_mask=partial_mask, model_kwargs=model_kwargs)
        
        img[partial_mask] = target[partial_mask]
        return img
    
    #for imputation or forecasting tasks
    # similar to p_sample but with the addition of the langevin dynamics and can be used for imputation tasks
    #input: x -> current state, t -> current timestep, target -> target, partial_mask -> mask for missing values
    #output: sample generated from the model
    def generate_noise(self, x, t):
        return torch.randn_like(x) if t > 0 else 0.

    def calculate_sigma(self, model_log_variance):
        return (0.5 * model_log_variance).exp()

    def generate_predicted_image(self, model_mean, sigma, noise):
        return model_mean + sigma * noise

    def update_predicted_image(self, pred_img, target_t, partial_mask):
        pred_img[partial_mask] = target_t[partial_mask]
        return pred_img

    def p_sample_infill(self, x, target, t: int, partial_mask=None, clip_denoised=True, model_kwargs=None):
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, _ = self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised)

        noise = self.generate_noise(x, t)
        sigma = self.calculate_sigma(model_log_variance)
        pred_img = self.generate_predicted_image(model_mean, sigma, noise)

        pred_img = self.langevin_func(sample=pred_img, mean=model_mean, sigma=sigma, t=batched_times, tgt_embs=target, partial_mask=partial_mask, **model_kwargs)

        target_t = self.q_sample(target, t=batched_times)
        pred_img = self.update_predicted_image(pred_img, target_t, partial_mask)

        return pred_img

    #refines the sample generated by the model using langevin dynamics
    #input: sample -> sample generated by the model, mean -> mean of the posterior distribution, sigma -> standard deviation of the posterior distribution, t -> current timestep, tgt_embs -> target, partial_mask -> mask for missing values
    #output: refined sample
    def adjust_learning_rate_and_iterations(self, t, learning_rate):
        # current timestep is less than 5% of the total timesteps, return 0 iterations and the current learning rate
        if t[0].item() < self.num_timesteps * 0.05:
            return 0, learning_rate
        # current timestep is more than 90% of the total timesteps, return 3 iterations and the current learning rate
        elif t[0].item() > self.num_timesteps * 0.9:
            return 3, learning_rate
        # current timestep is more than 75% of the total timesteps, return 2 iterations and half the current learning rate
        elif t[0].item() > self.num_timesteps * 0.75:
            return 2, learning_rate * 0.5
        # other cases, return 1 iteration and a quarter of the current learning rate
        else:
            return 1, learning_rate * 0.25
    
    # actual learning process
    def langevin_func(self, coef, partial_mask, tgt_embs, learning_rate, sample, mean, sigma, t, coef_=0.):
        K, learning_rate = self.adjust_learning_rate_and_iterations(t, learning_rate)

        input_embs_param = torch.nn.Parameter(sample)

        for _ in range(K):
            # enable gradient computation and initialize the optimizer
            with torch.enable_grad(), torch.optim.Adagrad([input_embs_param], lr=learning_rate) as optimizer:
                # compute the output embeddings at the current timestep
                x_start = self.output(x=input_embs_param, t=t)
                
                #case if mean is 0
                sigma_mean = sigma.mean()
                sigma_value = 1. if sigma_mean == 0 else sigma_mean

                #log probability term of the loss function
                logp_term = coef * ((mean - input_embs_param)**2 / sigma_value).mean(dim=0).sum()
                infill_loss = ((x_start[partial_mask] - tgt_embs[partial_mask]) ** 2 / sigma_value).mean(dim=0).sum()

                loss = logp_term + infill_loss
                loss.backward()
                optimizer.step()

                epsilon = torch.randn_like(input_embs_param.data)
                input_embs_param.data.add_(coef_ * sigma_mean.item() * epsilon)

        sample[~partial_mask] = input_embs_param.data[~partial_mask]
        return sample
    

if __name__ == '__main__':
    pass

