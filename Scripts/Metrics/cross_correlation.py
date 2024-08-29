# Description: This script contains the implementation of the cross-correlation loss function.
import torch
import numpy as np
from torch import nn

def random_choice(size, num_select=100):
    select_idx = np.random.randint(low=0, high=size, size=(num_select,))
    return select_idx

def cross_autocorrelation_fun(x, max_lag, dim=(0, 1)):
    def get_lower_triangular_indices(n):
        indices = torch.tril_indices(n, n)
        return [list(index) for index in indices]

    indices = get_lower_triangular_indices(x.shape[2])
    normalized_x = (x - x.mean(dim, keepdims=True)) / x.std(dim, keepdims=True)
    x_l = normalized_x[..., indices[0]]
    x_r = normalized_x[..., indices[1]]
    cacf_list = []

    for i in range(max_lag):
        if i > 0:
            y = x_l[:, i:] * x_r[:, :-i]
        else:
            y = x_l * x_r
        cacf_i = torch.mean(y, (1))
        cacf_list.append(cacf_i)

    cacf = torch.cat(cacf_list, 1)
    reshaped_cacf = cacf.reshape(cacf.shape[0], -1, len(indices[0]))

    return reshaped_cacf


class Loss(nn.Module):
    def __init__(self, name, regularization=1.0, transform=lambda x: x, threshold=10., backward=False, norm_function=lambda x: x):
        super(Loss, self).__init__()
        self.name = name
        self.regularization = regularization
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_function = norm_function

    def forward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)
        return self.regularization * self.loss_componentwise.mean()

    def compute(self, x_fake):
        raise NotImplementedError()

    @property
    def success(self):
        return torch.all(self.loss_componentwise <= self.threshold)


class CrossCLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(CrossCLoss, self).__init__(norm_function=lambda x: torch.abs(x).sum(0), **kwargs)
        self.cross_correl_real = self._calculate_cross_correlation(x_real)

    def _calculate_cross_correlation(self, x):
        transformed_x = self.transform(x)
        cross_correlation = cross_autocorrelation_fun(transformed_x, 1).mean(0)[0]
        return cross_correlation

    def compute(self, x_fake):
        cross_correl_fake = self._calculate_cross_correlation(x_fake)
        loss = self.norm_function(cross_correl_fake - self.cross_correl_real.to(x_fake.device))
        return loss / 10.