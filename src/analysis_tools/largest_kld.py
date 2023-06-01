import os
import torch
import numpy as np
from torchvision.utils import save_image
from src.seed import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def comparing_baseline(dataset, model, loss_fn, save_file, args, batch_size=64):
    path = os.path.join(args.output_dir, args.model_type, save_file)

    model.zero_grad()
    set_seed(args)
    composition_index = np.random.choice(
        len(dataset.data),
        10,
    )
    imgs = []
    for idx in range(len(composition_index)):
        imgs.append(dataset.__getitem__(composition_index[idx])[0])
    imgs = torch.stack(imgs, dim=0).to(next(model.parameters()).device)

    encoder_output = None
    if "group" in args.model_type:
        _, encoder_output, _, group_action, attn_dist, sub_symmetries = model(
            imgs, loss_fn
        )

    else:
        _, encoder_output, _ = model(
            imgs, loss_fn
        )  # ((loss), (encoder_output), (decoder_output))

    # KLD(z_1||z_2)
    slice = encoder_output[0].size(0) // 2
    mu_1, mu_2 = encoder_output[1][:slice], encoder_output[1][slice:]
    logvar_1, logvar_2 = torch.exp(encoder_output[2][:slice]), torch.exp(
        encoder_output[2][slice:]
    )
    z1, z2 = encoder_output[0][:slice], encoder_output[0][slice:]  # (slice, 1, dim)

    kld_z = (
        0.5 * torch.log(logvar_2 / logvar_1)
        + (logvar_1 + (mu_1 - mu_2) ** 2) / logvar_2 / 2
    )  # (slice, dim)
    top_k = None
    if args.dataset == "shapes3d":
        top_k = 6
    else:
        top_k = 5
    _, idx = torch.topk(kld_z, top_k, dim=-1)  # (slice, top_k)

    transformed_z1, transformed_z2 = [z1], [z2]
    row, col = idx.size()

    for i in range(col):
        temp = transformed_z1[i].clone()
        for j in range(row):
            temp[j, idx[j, i]] = z2[j, idx[j, i]]
        transformed_z1.append(temp)
    transformed_z1.append(z2)
    transformed_z1 = torch.stack(transformed_z1, dim=0)  # (top_k+2, slice, dim)
    transformed_z1 = transformed_z1.transpose(1, 0)  # (slice, top_k+2, dim)
    dim = transformed_z1.size(-1)

    transforemd_z1 = transformed_z1.reshape(-1, dim)  # (slice * (top_k+2), dim)

    output = model.decoder(transforemd_z1)
    save_dir = os.path.join(path, "comparing_interval" + ".png")
    save_image(output[0], save_dir, nrow=top_k + 2, pad_value=1.0)

    return
