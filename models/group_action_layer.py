import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Group_Elements(nn.Module):
    def __init__(self, config):
        super(Group_Elements, self).__init__()

        self.dim = config.latent_dim
        self.sec = self.dim
        self.sub_sec = 16

        self.num_elements = config.latent_dim * self.sub_sec

        self.group_elements = nn.Parameter(
            torch.Tensor(self.sec * self.sub_sec, config.latent_dim, config.latent_dim)
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=4, stride=2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=4, stride=2),
            nn.ReLU(True),
        )
        self.linear = nn.Sequential(
            nn.Linear(4 * self.dim, (self.sub_sec + 2) * self.sec)
        )
        self.th = config.th

    def forward(self, mean, logvar, latent_z):
        batch, _ = latent_z.size()
        slice = batch // 2

        mean_1, mean_2 = mean[:slice], mean[slice:]
        std_1, std_2 = torch.exp(0.5 * logvar[:slice]), torch.exp(0.5 * mean[slice:])

        latent_z1, latent_z2 = latent_z[:slice], latent_z[slice:]
        orthogonal_loss, parallel_loss, commut_loss, sparse_loss = self.indepdent_loss(
            latent_z
        )
        z_difference = latent_z1 - latent_z2
        (
            group_action,
            inverse_action,
            attn,
            sector_loss,
            sub_symmetries,
        ) = self.extract_group_action(
            mean_1, std_1, mean_2, std_2, z_difference
        )  # [batch/2, dim, dim]

        # disconnet latent zs grad graph to prevent each zs are closed.
        transformed_latent_z1 = torch.bmm(
            latent_z1.unsqueeze(1), group_action
        ).squeeze()  # [batch/2, dim]
        transformed_latent_z2 = torch.bmm(
            latent_z2.unsqueeze(1), inverse_action
        ).squeeze()  # [batch/2, dim]
        difference1 = F.mse_loss(transformed_latent_z2, latent_z1)
        difference2 = F.mse_loss(transformed_latent_z1, latent_z2)

        equivariant = difference1 + difference2  # + sub_equiv_loss

        return (
            transformed_latent_z1,
            transformed_latent_z2,
            group_action,
            equivariant,
            attn,
            orthogonal_loss,
            parallel_loss,
            commut_loss,
            sparse_loss,
            sector_loss,
            sub_symmetries,
        )

    def indepdent_loss(self, mean):
        batch = mean.size(0)
        symmetries = torch.matrix_exp(self.group_elements)
        transformed_mean = torch.matmul(mean, symmetries)  # (# of basis, |B|, |D|)

        difference_mean = mean - transformed_mean  # (# of basis, |B|, |D|)
        difference_mean_square = (
            difference_mean.reshape(-1, self.dim) ** 2
        )  # (# of basis, |B|, |D|) --> (# of basis * |B|, |D|)
        sum_square = difference_mean_square.sum(-1)  # (# of basis * |B|)
        target = torch.max(difference_mean_square, dim=-1).values  # (# of basis * |B|)
        spars_loss = F.mse_loss(sum_square, target)
        difference_mean = difference_mean.reshape(
            self.sec, self.sub_sec, -1, self.dim
        )  # (|S|, |S.S|, |B|, |D|)

        # parallel loss
        difference_mean_par = difference_mean.reshape(
            self.sec, -1, self.dim
        )  # (|S|, |S.S| * |B|, |D|)
        par_norm = torch.norm(difference_mean_par, dim=-1, keepdim=True)
        par_norm = torch.bmm(par_norm, par_norm.transpose(-1, -2))
        difference_mean_par = torch.bmm(
            difference_mean_par, difference_mean_par.transpose(-1, -2)
        )  # (|S|, |S.S|*|B|, |S.S|*|B|)
        difference_mean_par = -torch.log((difference_mean_par / par_norm) ** 2 + 1e-9)
        parallel_loss = difference_mean_par.mean()

        # orthogonal loss
        idx = np.random.choice(self.sec, self.sec, replace=True)  # |S.S| & |S|
        selected_mean = []
        for i in range(self.sec):
            selected_mean.append(difference_mean[i, idx[i], :, :])  # (|B|, |D|)
        selected_mean = torch.cat(selected_mean, dim=0)  # (|S|*|B|, |D|)
        selec_norm = torch.norm(selected_mean, dim=-1, keepdim=True)
        selec_norm = torch.mm(
            selec_norm, selec_norm.transpose(-1, -2)
        )  # (|S|*|B|, |S|*|B|)
        selected_mean = torch.mm(
            selected_mean, selected_mean.transpose(-1, -2)
        )  # (|S|*|B|, |S|*|B|)
        selected_mean = (selected_mean / selec_norm) ** 2  # (|S|*|B|, |S|*|B|)
        mask = self.block_diagonal_mask(selected_mean, self.sec)

        selected_mean = selected_mean * mask
        orthogonal_loss = selected_mean.mean()
        commut_loss = self.commutative(self.dim)

        return (
            orthogonal_loss,
            parallel_loss,
            commut_loss,
            spars_loss,
        )  # sub_loss1 + sub_loss2 + commut_loss

    def block_diagonal_mask(self, transformed_z_dot_matrix, sub_dim_size):
        mask = torch.ones_like(
            transformed_z_dot_matrix
        )  # (|B| * # of basis, |B|/2 * # of basis)
        dim = mask.size(0)
        section = dim // sub_dim_size
        assert dim % sub_dim_size == 0

        for i in range(sub_dim_size):
            mask[i * section : (i + 1) * section, i * section : (i + 1) * section] = 0

        return mask

    def extract_group_action(
        self, mean1, std1, mean2, std2, z_difference
    ):  # latent_z1, latent_z2):
        group_action = None
        sub_batch = mean1.size(0)
        sector_target = (
            (torch.abs(z_difference) > self.th).to(torch.int64).reshape(-1)
        )  # (|B||D|/2,)
        difference = torch.cat([mean1, std1, mean2, std2], dim=-1)
        prob = self.linear(difference)  # (|B|/2, 1.2*# of basis)
        sector_prob = (
            prob[:, : 2 * self.sec].reshape(-1, self.sec, 2).reshape(-1, 2)
        )  # (|B|/2, 2*dim) --> (|B|/2, |D|, 2) --> (|B||D|/2, 2)
        sector_loss = (
            F.cross_entropy(
                torch.softmax(sector_prob, dim=-1), sector_target, reduction="sum"
            )
            / sub_batch
        )
        factor_prob = prob[:, 2 * self.sec :]

        # get switch (off or on)
        batch_mul_dim = sector_prob.size(0)
        switch = torch.zeros(size=(batch_mul_dim,)).to(prob.device)
        sector_attn = F.gumbel_softmax(sector_prob, tau=0.0001)

        # find off switch
        boolian1 = sector_attn[:, 0] >= 0.5  # (|B|/2, 2)
        indexs1 = boolian1.nonzero(as_tuple=False).squeeze()
        switch[indexs1] = sector_attn[indexs1, 1]

        # find on switch
        boolian2 = sector_attn[:, 1] > 0.5  # (|B|/2, 2)
        indexs2 = boolian2.nonzero(as_tuple=False).squeeze()
        switch[indexs2] = sector_attn[indexs2, 1]
        switch = (
            switch.unsqueeze(1).transpose(-1, -2).reshape(-1, self.sec)
        )  # (|B||D|/2,) --> (|B||D|/2, 1) --> (1, |B||D|/2) --> (|B|, |D|/2)
        factor_prob = torch.softmax(
            factor_prob.view(-1, self.sec, self.sub_sec), dim=-1
        ).view(-1, self.num_elements)
        sector_weight = switch.repeat_interleave(
            self.sub_sec, dim=-1
        )  # prob[:, :self.dim].repeat_interleave(self.dim, dim=-1) # (|B|/2, |D|) --> (|B|/2, #of basis)
        weight = sector_weight * factor_prob  # (|B|/2, # of basis)
        weight = weight.unsqueeze(-1).unsqueeze(-1)  # (|B|/2, # of basis, 1, 1)

        symmetries = weight * self.group_elements  # (|B|/2, # of basis, dim, dim)
        sub_symmetries = symmetries.reshape(
            -1, self.sec, self.sub_sec, self.dim, self.dim
        )
        sub_symmetries = sub_symmetries.sum(
            2
        )  # (|B|/2 , # of section, # of sub symmetries in section, dim, dim) --> (|B|/2, # of section, dim, dim)
        symmetries = symmetries.sum(
            1
        )  # (|B|/2, # of basis, dim, dim) --> (|B|/2, dim, dim)
        forward_syms = torch.matrix_exp(symmetries)
        inverse_syms = torch.matrix_exp(-symmetries)
        attn_score = torch.cat(
            [switch, factor_prob], dim=-1
        )  # prob.squeeze().squeeze() # (|B|/2, 1.1*# of basis, 1, 1) --> (|B|/2, # of basis)
        return (
            forward_syms,
            inverse_syms,
            attn_score,
            sector_loss,
            sub_symmetries,
        )  # [batch/2, dim, dim], [batch/2, dim * k]

    def commutative(self, div=1):
        loss = 0.0

        basis = self.group_elements.reshape(
            -1, self.dim * self.num_elements
        )  # (|D| * # of basis, |D|)
        basis_matrix = torch.matmul(
            self.group_elements, basis
        )  # (# of basis, |D|, |D| * # of basis)
        basis_matrix = basis_matrix.view(-1, self.dim * self.num_elements)
        basis_matrix_t = basis_matrix.clone()
        for i in range(self.num_elements // div):
            for j in range(i + 1, self.num_elements // div, 1):
                basis_matrix_t[
                    j * self.dim : (j + 1) * self.dim, i * self.dim : (i + 1) * self.dim
                ] = basis_matrix[
                    i * self.dim : (i + 1) * self.dim, j * self.dim : (j + 1) * self.dim
                ]
                basis_matrix_t[
                    i * self.dim : (i + 1) * self.dim, j * self.dim : (j + 1) * self.dim
                ] = basis_matrix[
                    j * self.dim : (j + 1) * self.dim, i * self.dim : (i + 1) * self.dim
                ]

                difference = (basis_matrix - basis_matrix_t) ** 2
                loss = loss + difference.mean()
        return loss

    # for CKA loss
    def cka(self, latent_z, transformed_z):
        # latent_z: [batch, latent_dim], transformed_z: [latent_dim, batch, latent_dim]
        # cross-covariance [batch, latent_dim] --> [batch, batch]
        cross_cov_z = torch.mm(latent_z, latent_z.transpose(-1, -2))  # [batch, batch]
        if transformed_z.ndim >= 3:
            cross_cov_trans_z = torch.bmm(
                transformed_z, transformed_z.transpose(-1, -2)
            )  # [latent_dim, batch, batch]
        else:
            cross_cov_trans_z = torch.mm(transformed_z, transformed_z.transpose(-1, -2))

        batch = cross_cov_z.size(0)
        centering_matrix = (
            torch.eye(batch) - 1 / batch * torch.ones(size=(batch, batch))
        ).to(cross_cov_z.device)

        hsic_kl = self.hsic(
            cross_cov_z, cross_cov_trans_z, centering_matrix
        )  # k: cross_cov_z, l: cross_cov_trans_z
        hsic_kk = self.hsic(cross_cov_z, cross_cov_z, centering_matrix)
        hsic_ll = self.hsic(cross_cov_trans_z, cross_cov_trans_z, centering_matrix)

        cka = hsic_kl / torch.sqrt(hsic_kk * hsic_ll)  # 1-D tensor [latent_dim]
        return cka

    def hsic(self, cross_cov_z, cross_cov_trans_z, centering_matrix):
        if cross_cov_z.ndim >= 3:
            kh = torch.matmul(cross_cov_trans_z, centering_matrix)
        else:
            kh = torch.mm(cross_cov_z, centering_matrix)  # [batch, batch]
        lh = torch.matmul(
            cross_cov_trans_z, centering_matrix
        )  # [latent_dim, batch, batch]
        hsic = torch.matmul(kh, lh)  # [latent_dim, batch, batch]
        if hsic.ndim >= 3:
            dim, batch = lh.size(0), lh.size(1)
            trace = []
            for i in range(dim):
                trace.append(torch.trace(hsic[i]))
            trace = torch.stack(trace)  # 1-D tensor
            trace = 1 / ((batch - 1) ** 2) * trace
        else:
            batch = kh.size(0)
            trace = torch.trace(hsic)  # []
            trace = 1 / ((batch - 1) ** 2) * trace
        return trace  # hsic

    def init_weights(self):
        for n, p in self.named_parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
