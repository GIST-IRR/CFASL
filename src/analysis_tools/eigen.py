import os
import torch
import numpy as np
from tqdm import tqdm
from src.seed import set_seed
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def analysis_of_basis(dataset, model, loss_fn, save_file, args, batch_size=64):
    set_seed(args)
    train_sampler = (
        RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    )
    train_dataloader = DataLoader(
        dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        drop_last=False,
        pin_memory=True,
    )
    global_step = 0

    iteration = tqdm(train_dataloader, desc="Iteration")

    stacked_z = []
    for i, (data, class_label) in enumerate(iteration):
        model.eval()
        data = data.to(device)
        outputs = model.encoder(data)
        stacked_z.append(outputs[0].detach().cpu())
        global_step += 1

        if global_step == 100:
            break

    stacked_z = torch.cat(stacked_z)  # (batch * global_step, dim)
    stacked_z = stacked_z.transpose(-1, -2).numpy()  # (di, batch * global_step)

    cov_mat = np.cov(stacked_z)  # (dim, dim)
    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)[0], np.linalg.eig(cov_mat)[1]

    sorted_idx = np.argsort(eigenvalues)[::-1]  # desending
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[sorted_idx]

    # save eigenvalues graph
    path = os.path.join(args.output_dir, args.model_type, save_file)
    save_dir = os.path.join(path, "eigenvalues" + ".png")
    # bar chart
    fig, ax = plt.subplots(figsize=(10, 10))

    x = np.arange(cov_mat.shape[1])
    x_label = []
    for i in x:
        x_label.append("eig_v" + str(i))
    ax.bar(x, eigenvalues)
    ax.set_xticks(x, x_label)
    plt.savefig(save_dir)
    plt.close()

    # save eigenvectors
    path = os.path.join(args.output_dir, args.model_type, save_file)
    save_dir = os.path.join(path, "eigenvectors" + ".png")
    # heatmap
    plt.pcolor(np.abs(eigenvectors))
    plt.colorbar()
    plt.savefig(save_dir)
    plt.close()

    print(np.mean(np.abs(eigenvectors)))
    print(np.std(np.abs(eigenvectors)))
    return
