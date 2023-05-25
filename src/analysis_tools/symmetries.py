import os
import pickle
import torch
import numpy as np
from torchvision.utils import save_image
from src.analysis_tools.utils import fixed_one_factor
from src.disent_metrics.betavae import find_index_from_factors

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def extract_group_actions(dataset, model, loss_fn, save_file, args, batch_size=64):
    # fixed_indexs = np.array([i for i in range(dataset.factor_num)])
    fixed_indexs = [i for i in range(dataset.factor_num)]

    # Sample two mini batches of latent variables.
    num_factor, dim = dataset.factor_num, model.group_action_layer.dim
    sec, sub_sec = model.group_action_layer.sec, model.group_action_layer.sub_sec
    group_actions, attn_dists = np.zeros(shape=(num_factor, dim, dim)), np.zeros(
        shape=(num_factor, sec * (sub_sec + 1)))  # np.zeros(shape=(num_factor, (dim+1)*dim))
    difference = 0.0

    random_idx = np.random.choice(len(dataset.data), 1, )
    factor_base = np.squeeze(dataset.latents_classes[random_idx])
    # Set the factor value to increase by 1.
    for i in range(len(dataset.factor_dict)):
        input_1, input_2 = [], []
        for j in range(dataset.factor_dict[i] - 1):
            sub_input_1 = np.copy(factor_base)
            sub_input_2 = np.copy(factor_base)

            sub_input_1[i] = j
            sub_input_2[i] = j + 1

            input_1.append(sub_input_1)
            input_2.append(sub_input_2)

            if args.dataset == "car" and j > 25:
                break

        input_1_idx = find_index_from_factors(input_1, dataset)
        input_2_idx = find_index_from_factors(input_2, dataset)

        imgs_1, imgs_2 = [], []
        for k in range(len(input_1_idx)):
            imgs_1.append(dataset.__getitem__(input_1_idx[k])[0])
            imgs_2.append(dataset.__getitem__(input_2_idx[k])[0])
        imgs_1 = torch.stack(imgs_1, dim=0).to(next(model.parameters()).device)
        imgs_2 = torch.stack(imgs_2, dim=0).to(next(model.parameters()).device)

        imgs = torch.cat([imgs_1, imgs_2], dim=0)
        _, encoder_output, _, group_action, attn_dist, _ = model(imgs, loss_fn)

        #
        interval_size = group_action.size(0)
        transformed_z1 = torch.bmm(encoder_output[0][:interval_size].unsqueeze(1), group_action).squeeze()
        transformed_z1 = torch.cat([encoder_output[0][0].unsqueeze(0), transformed_z1], dim=0)

        #
        transformed_z2 = [encoder_output[0][0].unsqueeze(0)]
        for l in range(interval_size):
            t_z = torch.mm(transformed_z2[l], group_action[l])
            transformed_z2.append(t_z)
        transformed_z2 = torch.cat(transformed_z2, dim=0)
        group_action = group_action.mean(0)
        attn_dist = attn_dist.mean(0)

        # if j == 0:
        group_actions[i] = group_action.detach().cpu().numpy()
        attn_dists[i] = attn_dist.detach().cpu().numpy()

        output_1 = model.decoder(transformed_z1)
        output_2 = model.decoder(transformed_z2)

        target_img = torch.cat([imgs_1[0].unsqueeze(0), imgs_2])
        n_row = target_img.size(0)
        path = os.path.join(args.output_dir, args.model_type, save_file)
        save_dir = os.path.join(path, 'transformed_' + str(i) + '.png')
        save_image(torch.cat([target_img, output_1[0]]), save_dir, nrow=n_row, pad_value=1.0)
        save_dir = os.path.join(path, 'interval_' + str(i) + '.png')
        save_image(torch.cat([target_img, output_2[0]]), save_dir, nrow=n_row, pad_value=1.0)

        model.zero_grad()
        #
        composition_index = np.random.choice(len(dataset.data), 10, )
        imgs = []
        for idx in range(len(composition_index)):
            imgs.append(dataset.__getitem__(composition_index[idx])[0])
        imgs = torch.stack(imgs, dim=0).to(next(model.parameters()).device)
        _, encoder_output, _, group_action, attn_dist, sub_symmetries = model(imgs, loss_fn)
        transformed_z3 = torch.bmm(encoder_output[0][:5].unsqueeze(1), group_action).squeeze()
        input_3 = torch.cat([encoder_output[0][:5], transformed_z3], dim=0)
        output_3 = model.decoder(input_3)
        save_dir = os.path.join(path, 'composition' + '.png')

        order_list = torch.Tensor([i for i in range(10)]).reshape(2, -1).transpose(-1, -2).reshape(-1).numpy()
        save_image(torch.cat([imgs[order_list], output_3[0][order_list]]), save_dir, nrow=args.interval, pad_value=1.0)

        #
        dim = encoder_output[0][:5].size(-1)
        transformed_z4 = [encoder_output[0][:5].unsqueeze(1)]  # (5, 1, dim)
        for i in range(dim):
            t_z = torch.bmm(transformed_z4[0], torch.matrix_exp(sub_symmetries[:, 0:i + 1, :, :].sum(1)))  # (5, 1, dim)
            transformed_z4.append(t_z)
        transformed_z4 = torch.cat(transformed_z4, dim=1).reshape(-1, dim)  # (5, # of sub symmetries + 1, dim) --> ( 5 * (# of sub symmetries + 1), dim
        output_4 = model.decoder(transformed_z4)
        save_dir = os.path.join(path, 'composition_interval' + '.png')
        save_image(output_4[0], save_dir, nrow=dim + 1, pad_value=1.0)
        model.zero_grad()

    group_actions = group_actions / 10
    attn_dists = attn_dists / 10
    print(difference)

    path = os.path.join(args.output_dir, args.model_type, save_file)
    with open(os.path.join(path, 'group_action.pickle'), 'wb') as f:
        pickle.dump(group_actions, f)
    with open(os.path.join(path, 'attn_dist.pickle'), 'wb') as f:
        pickle.dump(attn_dists, f)

    return