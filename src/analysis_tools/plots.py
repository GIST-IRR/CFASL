import os
import pickle
import torch
import numpy as np
from src.analysis_tools.utils import find_index_from_factors
from src.disent_metrics.betavae import find_index_from_factors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plotting_3d(dataset, model, loss_fn, save_file, args, batch_size=64):
    z_list = []
    class_list = []
    fixed_factor_value = {}  # key: idx, value: value
    # updates fixed_factor_value dictionary
    # Fixed value during sampling of the factor index to be fixed for each data set
    # dataste.factor_dict = {0:3, 1:6, 2:40, 3:32, 4:32} (dsprites)
    if args.dataset == "dsprites":
        fixed_factor_value[1] = np.random.choice(dataset.factor_dict[1], 1)
        fixed_factor_value[2] = np.random.choice(dataset.factor_dict[2], 1)
    if args.dataset == "shapes3d":
        fixed_factor_value[2] = np.random.choice(dataset.factor_dict[2], 1)
        fixed_factor_value[3] = np.random.choice(dataset.factor_dict[3], 1)
        fixed_factor_value[5] = np.random.choice(dataset.factor_dict[5], 1)
    if args.dataset == "smallnorb":
        fixed_factor_value[3] = np.random.choice(dataset.factor_dict[3], 1)
    # A total of 10 random sampling
    # mini-batch size = 64
    for i in range(10):
        # Random sampling of mini-batch size
        composition_index = np.random.choice(len(dataset.data), batch_size)
        imgs, factors = [], []
        # Find the Factor Label corresponding to the Dataset Index and save it to the Factors List
        for idx in composition_index:
            factors.append(dataset.latents_classes[idx])
        factors = np.stack(factors, axis=0)  # (batch, # of factors)

        # fix the specific factors value
        for key, value in fixed_factor_value.items():
            factors[:, key] = value

        # add class lables of each input for coloring (plot)
        if args.dataset == "dsprites" or args.dataset == "smallnorb":
            class_list.append(factors[:, 0])
        if args.dataset == "shapes3d":
            class_list.append(factors[:, 4])
        # Find the image index with a fixed factor label and select it as an input
        idx = find_index_from_factors(factors, dataset)
        for id in idx:
            imgs.append(dataset.__getitem__(id)[0])
        imgs = torch.stack(imgs, dim=0).to(next(model.parameters()).device)

        # run model
        outputs = model.encoder(imgs)
        z, mu, logvar = outputs[0], outputs[1], outputs[2]
        kld = -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp(), dim=0)
        # pick highest 3 dim values.
        _, topk_idx = torch.topk(kld, 3, dim=-1)
        z_list.append(z[:, topk_idx].detach().cpu().numpy())

    z_list = np.concatenate(z_list, axis=0)  # (10 * batch, dim)
    class_list = np.concatenate(class_list)  # (10 * batch)

    path = os.path.join(args.output_dir, args.model_type, save_file)
    with open(os.path.join(path, "plot_3d.pickle"), "wb") as f:
        pickle.dump(z_list, f)
    with open(os.path.join(path, "plot_labels.pickle"), "wb") as f:
        pickle.dump(class_list, f)

    return
