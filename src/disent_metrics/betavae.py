import logging
import torch
import numpy as np
from src.seed import set_seed
from sklearn import linear_model

logger = logging.getLogger(__name__)
def compute_beta_vae(dataset, model, batch_size, num_train, num_eval, loss_fn, args):
    logger.info("*********************Beta-VAE Disentanglement Evaluation*********************")

    train_points, train_labels = generate_training_batch(dataset, model, batch_size, num_train, loss_fn, args)
    set_seed(args)
    regression_model = linear_model.LogisticRegression(max_iter=200)
    regression_model.fit(train_points, train_labels)

    train_accuracy = regression_model.score(train_points, train_labels)
    logging.info("Training set accuracy: %.2g", train_accuracy)

    eval_points, eval_labels = generate_training_batch(dataset, model, batch_size, num_eval, loss_fn, args)
    eval_accuracy = regression_model.score(eval_points, eval_labels)
    return eval_accuracy

def generate_training_batch(dataset, model, batch_size, num_points, loss_fn, args):
    points = None # Dimensionality depends on the representation function.
    labels = np.zeros(num_points, dtype=np.int64)
    set_seed(args)
    for i in range(num_points):
        labels[i], feature_vector = generate_training_sample(dataset, model, batch_size, loss_fn, args)
        if points is None:
            points = np.zeros((num_points, feature_vector.shape[0]))
        points[i, :] = feature_vector
    return points, labels

def generate_training_sample(dataset, model, batch_size, loss_fn, args):

    # select random coordinate to keep fixed.
    fixed_index = np.random.randint(dataset.factor_num)
    # Sample two mini batches of latent variables.
    length = len(dataset)
    idx_1 = np.random.choice(length - 1, batch_size, replace=False)
    idx_2 = np.random.choice(length - 1, batch_size, replace=False)
    # Sample two mini batches of latent variables.
    factors_1 = dataset.latents_classes[idx_1]
    factors_2 = dataset.latents_classes[idx_2]
    # Ensure sampled coordinate is the same across pairs of samples.
    factors_2[:, fixed_index] = factors_1[:, fixed_index]
    # Select index from factors
    changed_idx = find_index_from_factors(factors_2, dataset)
    # Select imgaes from idx
    imgs_1, imgs_2 = [], []
    for id in idx_1:
        imgs_1.append(dataset.__getitem__(id)[0])
    for id in changed_idx:
        imgs_2.append(dataset.__getitem__(id)[0])
    imgs_1 = torch.stack(imgs_1, dim=0).to(next(model.parameters()).device)
    imgs_2 = torch.stack(imgs_2, dim=0).to(next(model.parameters()).device)

    latent_vector_1 = model(imgs_1, loss_fn)[1][0]
    latent_vector_2 = model(imgs_2, loss_fn)[1][0]

    feature_vector = torch.mean(torch.abs(latent_vector_1 - latent_vector_2), dim=-2)
    feature_vector = feature_vector.detach().cpu().numpy() # (latent_dim)
    return fixed_index, feature_vector

def find_index_from_factors(factors, dataset):
    factor_dict = {}
    sampled_idx = []
    for i, classes in enumerate(dataset.latents_classes):
        factor_dict[classes.tobytes()] = i
    for factor in factors:
        sampled_idx.append(factor_dict[factor.tobytes()])
    return sampled_idx