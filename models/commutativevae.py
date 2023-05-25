import torch
import numpy as np
import torch.nn as nn
import math
from models.encoder import CNN2DShapesLieEncoder, CNN3DShapesLieEncoder
from models.decoder import CNN2DShapesLieDecoder, CNN3DShapesLieDecoder

class CommutativeVAE(nn.Module):
    def __init__(self, config):
        super(CommutativeVAE, self).__init__()
        self.hy_rec = config.hy_rec
        self.hy_hes = config.hy_hes
        self.hy_commute = config.hy_commute
        if config.dataset == 'dsprites' or config.dataset == 'smallnorb': # config.dataset == 'dsprites': # for 96 by 96 size
            self.encoder = CNN2DShapesLieEncoder(config)
            self.decoder = CNN2DShapesLieDecoder(config)
        else:
            self.encoder = CNN3DShapesLieEncoder(config)
            self.decoder = CNN3DShapesLieDecoder(config)
        self.subspace_sizes_ls = config.subspace_sizes_ls
        self.forward_eg_prob = config.forward_eq_prob
        self.subgroup_sizes_ls = config.subgroup_sizes_ls

    def forward(self, input, loss_fn):
        result = {'elbo': {}, 'obj': {}, 'id': {}}
        batch = input.size(0)
        #loss_fn = nn.BCELoss(reduction='sum')
        enc_output = self.encoder(input)
        z, mu, logvar, group_feats_E = enc_output[0], enc_output[1], enc_output[2], enc_output[3]
        output, group_feats_D = self.decoder(z)
        #pdb.set_trace()
        x_eg_hat = self.decoder.gfeat(group_feats_E)
        x_gg_hat = self.decoder.gfeat(group_feats_D)
        if self.training:
            rand_n = np.random.uniform()
            if rand_n < self.forward_eg_prob:
                rec_loss = loss_fn(x_eg_hat, input) / batch
            else:
                rec_loss = loss_fn(output, input) / batch
        else:
            rec_loss = loss_fn(output, input) / batch

        group_loss = self.group_loss(group_feats_E, group_feats_D, self.decoder.lie_alg_basis_ls)
        total_kl = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=-1))
        #beta_kl = self.self.control_capacity(total_kl, self.global_step, self.anneal)

        result['obj']['reconst'] = rec_loss.unsqueeze(0)
        result['obj']['kld'] = total_kl.unsqueeze(0)
        result['obj']['group'] = group_loss.unsqueeze(0)

        outputs = (result,) + (enc_output,) + (output,)
        return outputs

    def group_loss(self, group_feats_E, group_feats_G, lie_alg_basis_ls):
        b_idx = 0
        hessian_loss, commute_loss = 0., 0.

        for i, subspace_size in enumerate(self.subspace_sizes_ls):
            e_idx = b_idx + subspace_size
            if subspace_size > 1:
                mat_dim = int(math.sqrt(self.subgroup_sizes_ls[i]))
                assert list(lie_alg_basis_ls[b_idx].size())[-1] == mat_dim
                lie_alg_basis_mul_ij = self.calc_basis_mul_ij(lie_alg_basis_ls[b_idx:e_idx])
                hessian_loss += self.calc_hessian_loss(lie_alg_basis_mul_ij, i)
                commute_loss += self.calc_commute_loss(lie_alg_basis_mul_ij, i)
            b_idx = e_idx
        rec_loss = torch.mean(torch.sum(torch.square(group_feats_E - group_feats_G), dim=1)).unsqueeze(0)

        rec_loss *= self.hy_rec
        hessian_loss *= self.hy_hes
        commute_loss *= self.hy_commute
        loss = hessian_loss + commute_loss + rec_loss
        return loss

    def calc_basis_mul_ij(self, lie_alg_basis_ls_param):
        lie_alg_basis_ls = [alg_tmp * 1. for alg_tmp in lie_alg_basis_ls_param]
        lie_alg_basis = torch.cat(lie_alg_basis_ls,
                                  dim=0)[np.newaxis,
                                         ...]  # [1, lat_dim, mat_dim, mat_dim]
        _, lat_dim, mat_dim, _ = list(lie_alg_basis.size())
        lie_alg_basis_col = lie_alg_basis.view(lat_dim, 1, mat_dim, mat_dim)
        lie_alg_basis_outer_mul = torch.matmul(lie_alg_basis, lie_alg_basis_col)
        hessian_mask = 1. - torch.eye(lat_dim, dtype=lie_alg_basis_outer_mul.dtype)[:, :, np.newaxis, np.newaxis].to(lie_alg_basis_outer_mul.device)
        lie_alg_basis_mul_ij = lie_alg_basis_outer_mul * hessian_mask
        return lie_alg_basis_mul_ij

    def calc_hessian_loss(self, lie_alg_basis_mul_ij, i):
        hessian_loss = torch.mean(torch.sum(torch.square(lie_alg_basis_mul_ij), dim=[2,3]))
        return hessian_loss.unsqueeze(0)

    def calc_commute_loss(self, lie_alg_basis_mul_ij, i):
        lie_alg_commutator = lie_alg_basis_mul_ij - lie_alg_basis_mul_ij.permute(0,1,3,2)
        commute_loss = torch.mean(torch.sum(torch.square(lie_alg_commutator), dim=[2,3]))
        return commute_loss.unsqueeze(0)

    def init_weights(self):
        for n, p in self.named_parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            #else:
                #nn.init.zeros_(p.data)

