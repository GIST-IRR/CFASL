import torch
import math
import logging
import numpy as np
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm
from src.seed import set_seed

logger = logging.getLogger(__name__)
transform = ToTensor()


def mFVM(dataset, model, batch_size, num_train, loss_fn, fixed_idx_num, args):
    logger.info(
        "*********************Multi FVM Disentanglement Evaluation*********************"
    )
    model.eval()
    with torch.no_grad():
        length = len(dataset)
        idx = np.random.choice(length - 1, int(math.floor(0.1 * length)), replace=False)

        global_varaince = _compute_global_variance(dataset, model, batch_size, loss_fn)
        active_dims = _prune_dims(global_varaince)

        if not active_dims.any():
            return {"disentanglement_accuracy": 0.0, "num_active_dims": 0}

        votes = _generate_training_batch(
            dataset=dataset,
            model=model,
            batch_size=batch_size,
            num_points=num_train,
            variances=global_varaince,
            active_dims=active_dims,
            loss_fn=loss_fn,
            fixed_idx_num=fixed_idx_num,
            args=args,
        )
        accuracy = 0.0
        for main_key in votes.keys():
            count = []
            for key in votes[main_key]:
                count.append(votes[main_key][key])
            accuracy += np.max(np.array(count)) / np.sum(np.array(count))
        accuracy /= len(votes)
    return {
        "disentanglement_accuracy": np.asscalar(accuracy),
        "num_active_dims": active_dims.size(0),
    }


def _compute_global_variance(dataloader, model, batch_size, loss_fn):
    # compute variance of sampled dataset
    data = dataloader.random_sampling_for_disen_global_variance(batch_size)
    model.eval()
    with torch.no_grad():
        data = data.to(next(model.parameters()).device)
        z = model(data, loss_fn)[1][0]  # .squeeze(-1).squeeze(-1)
        var = torch.var(z, dim=-2)
    return var  # (dim,)


def _prune_dims(variances, threshold=0.06):
    """Mask for dimensions collapsed to the prior."""
    scale_z = torch.sqrt(variances)
    return scale_z >= threshold


def _generate_training_batch(
    dataset,
    model,
    batch_size,
    num_points,
    variances,
    active_dims,
    loss_fn,
    fixed_idx_num,
    args,
):
    votes = {}
    set_seed(args)
    for _ in tqdm(range(num_points), desc="Iteration"):
        factor_index, target = _generate_training_sample(
            dataset,
            model,
            batch_size,
            variances,
            active_dims,
            loss_fn,
            fixed_idx_num,
            args,
        )
        votes = update_votes_dict(votes, factor_index, target)
    return votes


def update_votes_dict(factor_dict, idx, target):  # dict, tuple, scalar
    if factor_dict == {}:
        factor_dict[idx] = {}
    else:
        temp_dict = factor_dict.copy()
        if idx not in temp_dict:
            factor_dict[idx] = {}
        # pdb.set_trace()
    if factor_dict[idx] == {}:
        factor_dict[idx][target] = 1
    else:
        temp_dict = factor_dict[idx].copy()
        if target not in temp_dict:
            factor_dict[idx][target] = 1
        else:
            factor_dict[idx][target] += 1

    return factor_dict


def _generate_training_sample(
    dataset, model, batch_size, variances, active_dims, loss_fn, fixed_idx_num, args
):
    length = len(dataset)
    idx = np.random.choice(length - 1, batch_size, replace=False)
    sampled_factors = dataset.latents_classes[idx]
    fixed_idx = np.random.choice(
        dataset.factor_num, fixed_idx_num, replace=False
    )  # ( |fixed_idx_num|,)
    sampled_factors[:, fixed_idx] = sampled_factors[0, fixed_idx]

    # find correspond idx of sampeld factors
    idx = find_index_from_factors(sampled_factors, dataset)

    # resize version
    resized_imgs = []
    for id in idx:
        resized_imgs.append(dataset.__getitem__(id)[0])
    observation = torch.stack(resized_imgs, dim=0).to(next(model.parameters()).device)

    # latent vectors
    z = model(observation, loss_fn)[1][0]
    var = torch.var(z, dim=-2)
    _, dim = torch.topk(
        var[active_dims] / variances[active_dims], fixed_idx_num, largest=False
    )  # torch.argmin(var[active_dims] / variances[active_dims])
    dim = dim.detach().cpu().numpy()
    return tuple(np.sort(fixed_idx)), tuple(np.sort(dim))


def find_index_from_factors(factors, dataset):
    factor_dict = {}
    sampled_idx = []
    for i, classes in enumerate(dataset.latents_classes):
        factor_dict[classes.tobytes()] = i
    for factor in factors:
        sampled_idx.append(factor_dict[factor.tobytes()])
    return sampled_idx
