"""
Based on "Disentangling by Factorising" (https://github.com/nmichlo/disent/blob/main/disent/metrics/utils.py).
"""
import numpy as np
import sklearn
import torch
from src.seed import set_seed
import pdb
def latents_and_factors(dataset, model, batch_size, interation, loss_fn, args):
    model.eval()
    with torch.no_grad():
        latents = []
        imgs, factors = dataset.sampling_factors_and_img(batch_size, interation)
        for img in imgs:
            img = img.to(next(model.parameters()).device)
            latent = model(img, loss_fn)[1][0]
            latents.append(latent.detach().cpu())
        latents = torch.cat(latents, dim=0).transpose(-1,-2).numpy() #(latent_dim, iteration*batch_size)
        factors = factors.view(interation * batch_size, -1).transpose(-1, -2).numpy() #(factor_dim, iteration*batch_size

    return latents, factors

def histogram_discretize(target, num_bins=20):
    """
    Discretization based on histograms.
    """
    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(target[i, :], np.histogram(target[i, :], num_bins)[1][:-1])
    return discretized

def discrete_mutual_info(mus, ys):
    """
    Compute discrete mutual information.
    """
    num_codes = mus.shape[0]
    num_factors = ys.shape[0]
    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):
            m[i, j] = sklearn.metrics.mutual_info_score(ys[j, :], mus[i, :])
    return m

def discrete_entropy(ys):
    """
    Compute discrete mutual information.
    """
    num_factors = ys.shape[0]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
    return h
