# Isolating Sources of Disentanglement in VAEs
"""
Based on "Disentangling by Factorising" (https://github.com/nmichlo/disent/blob/main/disent/metrics/_mig.py).
"""
import logging
import torch
import numpy as np
from src.disent_metrics import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


def MIGMetric(train_latents, train_factors):
    # model.eval()
    with torch.no_grad():
        logger.info(
            "************************MIG Disentanglement Evaluation************************"
        )
    return compute_mig(train_latents, train_factors)


def compute_mig(latent, factors):
    discreitezed_latent = utils.histogram_discretize(latent, num_bins=20)
    m = utils.discrete_mutual_info(discreitezed_latent, factors)
    assert m.shape[0] == latent.shape[0]
    assert m.shape[1] == factors.shape[0]
    entropy = utils.discrete_entropy(factors)
    sorted_m = np.sort(m, axis=0)[::-1]

    result = np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))
    return result
