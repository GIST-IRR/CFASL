import torch
import torch.nn as nn
import numpy as np
import math
from models.group_action_layer import Group_Elements
from models.encoder import CNN2DShapesLieEncoder, CNN3DShapesLieEncoder
from models.decoder import CNN2DShapesLieDecoder, CNN3DShapesLieDecoder


class CNNGroupCommutativeVAE(nn.Module):
    def __init__(self, config):
        super(CNNGroupCommutativeVAE, self).__init__()
        # self.prob = config.prob
        self.lamb = config.lamb
        self.hy_rec = config.hy_rec
        self.hy_hes = config.hy_hes
        self.hy_commute = config.hy_commute
        self.encoder = (
            CNN2DShapesLieEncoder(config)
            if config.dataset == "dsprites"
            else CNN3DShapesLieEncoder(config)
        )
        self.decoder = (
            CNN2DShapesLieDecoder(config)
            if config.dataset == "dsprites"
            else CNN3DShapesLieDecoder(config)
        )
        self.subspace_sizes_ls = config.subspace_sizes_ls
        self.forward_eg_prob = config.forward_eq_prob
        self.subgroup_sizes_ls = config.subgroup_sizes_ls

        self.group_action_layer = Group_Elements(config)

    def forward(self, input, loss_fn):
        # if self.training:
        encoder_output = self.encoder(
            input
        )  # (z, mu, logvar, (option: all_hidden_representation))
        batch, dim = encoder_output[0].size()
        slice = batch // 2

        (
            transformed_z1,
            transformed_z2,
            group_action,
            equivariant,
            attn,
            orthogonal_loss,
            parallel_loss,
            commut_loss,
            sparse_loss,
            sector_loss,
            sub_symmetries,
        ) = self.group_action_layer(
            encoder_output[1], encoder_output[2], encoder_output[0]
        )  # commutative_loss

        transformed_z = torch.cat(
            [transformed_z2, transformed_z1], dim=0
        )  # [batch, dim]
        merge_z = torch.cat(
            [encoder_output[0], transformed_z], dim=0
        )  # [2*batch, dim])

        decoder_output = self.decoder(encoder_output[0])
        outputs = (encoder_output,) + (decoder_output,)

        loss = self.loss(input, outputs, loss_fn)
        loss["obj"]["orthogonal"] = orthogonal_loss
        loss["obj"]["parallel"] = parallel_loss
        loss["obj"]["commutative"] = commut_loss
        loss["obj"]["sector"] = sector_loss
        loss["obj"]["sparse"] = sparse_loss
        # encoder equivariant loss
        loss["obj"]["equivariant_1"] = equivariant
        # decoder equivariant loss
        loss["obj"]["equivariant_2"] = loss_fn(outputs[1][0], input) / batch
        loss = (
            (loss,)
            + (encoder_output,)
            + (decoder_output,)
            + (group_action,)
            + (attn,)
            + (sub_symmetries,)
        )
        return loss

    def loss(self, input, outputs, loss_fn):
        result = {"elbo": {}, "obj": {}, "id": {}}
        batch = input.size(0)
        reconsted_images = outputs[1][0]  # [batch, c, h, w]
        group_feats_D = outputs[1][1]  # [batch, latent_dim]
        z, mu, logvar, group_feats_E = (
            outputs[0][0].squeeze(),
            outputs[0][1].squeeze(),
            outputs[0][2].squeeze(),
            outputs[0][3].squeeze(),
        )
        x_eg_hat = self.decoder.gfeat(group_feats_E)

        if self.training:
            rand_n = np.random.uniform()
            if rand_n < self.forward_eg_prob:
                rec_loss = loss_fn(x_eg_hat, input) / batch
            else:
                rec_loss = loss_fn(reconsted_images, input) / batch
        else:
            rec_loss = loss_fn(reconsted_images, input) / batch

        group_loss = self.group_loss(
            group_feats_E, group_feats_D, self.decoder.lie_alg_basis_ls
        )
        kld_err = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=-1)
        )

        result["obj"]["reconst"] = rec_loss.unsqueeze(0)
        result["obj"]["kld"] = kld_err.unsqueeze(0)
        result["obj"]["group"] = group_loss.unsqueeze(0)

        return result

    def group_loss(self, group_feats_E, group_feats_G, lie_alg_basis_ls):
        b_idx = 0
        hessian_loss, commute_loss = 0.0, 0.0

        for i, subspace_size in enumerate(self.subspace_sizes_ls):
            e_idx = b_idx + subspace_size
            if subspace_size > 1:
                mat_dim = int(math.sqrt(self.subgroup_sizes_ls[i]))
                assert list(lie_alg_basis_ls[b_idx].size())[-1] == mat_dim
                lie_alg_basis_mul_ij = self.calc_basis_mul_ij(
                    lie_alg_basis_ls[b_idx:e_idx]
                )
                hessian_loss += self.calc_hessian_loss(lie_alg_basis_mul_ij, i)
                commute_loss += self.calc_commute_loss(lie_alg_basis_mul_ij, i)
            b_idx = e_idx
        rec_loss = torch.mean(
            torch.sum(torch.square(group_feats_E - group_feats_G), dim=1)
        ).unsqueeze(0)

        rec_loss *= self.hy_rec
        hessian_loss *= self.hy_hes
        commute_loss *= self.hy_commute
        loss = hessian_loss + commute_loss + rec_loss
        return loss

    def calc_basis_mul_ij(self, lie_alg_basis_ls_param):
        lie_alg_basis_ls = [alg_tmp * 1.0 for alg_tmp in lie_alg_basis_ls_param]
        lie_alg_basis = torch.cat(lie_alg_basis_ls, dim=0)[
            np.newaxis, ...
        ]  # [1, lat_dim, mat_dim, mat_dim]
        _, lat_dim, mat_dim, _ = list(lie_alg_basis.size())
        lie_alg_basis_col = lie_alg_basis.view(lat_dim, 1, mat_dim, mat_dim)
        lie_alg_basis_outer_mul = torch.matmul(lie_alg_basis, lie_alg_basis_col)
        hessian_mask = 1.0 - torch.eye(lat_dim, dtype=lie_alg_basis_outer_mul.dtype)[
            :, :, np.newaxis, np.newaxis
        ].to(lie_alg_basis_outer_mul.device)
        lie_alg_basis_mul_ij = lie_alg_basis_outer_mul * hessian_mask
        return lie_alg_basis_mul_ij

    def calc_hessian_loss(self, lie_alg_basis_mul_ij, i):
        hessian_loss = torch.mean(
            torch.sum(torch.square(lie_alg_basis_mul_ij), dim=[2, 3])
        )
        return hessian_loss.unsqueeze(0)

    def calc_commute_loss(self, lie_alg_basis_mul_ij, i):
        lie_alg_commutator = lie_alg_basis_mul_ij - lie_alg_basis_mul_ij.permute(
            0, 1, 3, 2
        )
        commute_loss = torch.mean(
            torch.sum(torch.square(lie_alg_commutator), dim=[2, 3])
        )
        return commute_loss.unsqueeze(0)

    def init_weights(self):
        for n, p in self.named_parameters():
            if p.data.ndimension() >= 2 and "group_elements" in n:
                p.data = torch.normal(
                    mean=torch.zeros_like(p.data),
                    std=0.000001 * torch.ones_like(p.data),
                )  # nn.init.trunc_normal_(p.data)
            elif p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.uniform_(p.data)
