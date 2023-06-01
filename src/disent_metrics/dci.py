"""
Based on "Disentangling by Factorising" (https://github.com/nmichlo/disent/blob/main/disent/metrics/_dci.py).
"""
import logging
import scipy
import scipy.stats
import numpy as np
from src.seed import set_seed
from tqdm import tqdm

logger = logging.getLogger(__name__)


def metric_dci(
    train_latents,
    train_factors,
    test_latents,
    test_factors,
    args,
    show_progress=False,
    continuous_factors=False,
):
    logger.info(
        "*********************DCI Disentanglement Evaluation*********************"
    )
    scores = compute_dci(
        train_latents,
        train_factors,
        test_latents,
        test_factors,
        show_progress,
        continuous_factors,
        args,
    )

    return scores


def compute_dci(
    train_latents,
    train_factors,
    test_latents,
    test_factors,
    show_progress,
    continuous_factors,
    args,
):
    importance_matrix, train_err, test_err = compute_importance_gbt(
        train_latents,
        train_factors,
        test_latents,
        test_factors,
        show_progress,
        continuous_factors,
        args,
    )
    assert importance_matrix.shape[0] == train_latents.shape[0]
    assert importance_matrix.shape[1] == train_factors.shape[0]

    disentanglement = disentangle(importance_matrix)
    completeness = complete(importance_matrix)

    return train_err, test_err, disentanglement, completeness


def compute_importance_gbt(
    train_latents,
    train_factors,
    test_latents,
    test_factors,
    show_progress,
    continuous_factors,
    args,
):
    num_factors = train_factors.shape[0]
    num_latents = train_latents.shape[0]
    importance_matrix = np.zeros(shape=[num_latents, num_factors], dtype=np.float64)
    train_loss, test_loss = [], []
    for i in tqdm(range(num_factors)):
        # if mode == 'sklearn':
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.ensemble import GradientBoostingRegressor

        set_seed(args)
        model = (
            GradientBoostingRegressor()
            if continuous_factors
            else GradientBoostingClassifier()
        )

        model.fit(train_latents.T, train_factors[i, :])
        importance_matrix[:, i] = np.abs(model.feature_importances_)
        train_loss.append(
            np.mean(model.predict(train_latents.T) == train_factors[i, :])
        )
        test_loss.append(np.mean(model.predict(test_latents.T) == test_factors[i, :]))

    return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def disentangle(importance_matrix):
    per_code = disentanglement_per_code(importance_matrix)
    if importance_matrix.sum() == 0.0:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()
    return np.sum(per_code * code_importance)


def disentanglement_per_code(importance_matrix):
    return 1.0 - scipy.stats.entropy(
        importance_matrix.T + 1e-11, base=importance_matrix.shape[1]
    )


def complete(importance_matrix):
    per_factor = completeness_per_factor(importance_matrix)
    if importance_matrix.sum() == 0.0:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    return np.sum(per_factor * factor_importance)


def completeness_per_factor(importance_matrix):
    # (latents_dim, factors_dim)
    return 1.0 - scipy.stats.entropy(
        importance_matrix + 1e-11, base=importance_matrix.shape[0]
    )
