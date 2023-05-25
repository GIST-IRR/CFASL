import torch
import torch.nn as nn
import math
import numpy as np
from models.utils import CNNTrasnposedLayer


class CNN2DShapesDecoder(nn.Module):
    def __init__(self, config):
        super(CNN2DShapesDecoder, self).__init__()
        modules = []
        self.latent_dim = config.latent_dim
        self.hidden_states = config.hidden_states
        self.config_name = config.__class__.__name__
        self.num_inv_equ = config.num_inv_equ if 'CHIC' in self.config_name else None#self.config_name =='CNNCHICConfig' else None
        # Design Decoder Factor-VAE ref
        self.dense1 = nn.Linear(self.latent_dim, config.dense_dim[1])
        self.dense2 = nn.Linear(config.dense_dim[1], 4*config.dense_dim[0])
        self.relu = nn.ReLU(True)
        self.active = nn.Sigmoid()

        modules.append(CNNTrasnposedLayer(in_channels=64, out_channels=64))
        modules.append(CNNTrasnposedLayer(in_channels=64, out_channels=32))
        modules.append(CNNTrasnposedLayer(in_channels=32, out_channels=32))
        modules.append(nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1))
        self.hidden_layers = nn.ModuleList(modules)

    def forward(self, input):
        # input: [batch, latent_dim]
        all_hidden_states = ()
        output = self.dense1(input)
        output = self.relu(output)
        output = self.dense2(output)
        output = self.relu(output) # (B, ...)
        if 'CHIC' in self.config_name: #!= 'CNNCHICConfig':
            output = output.view(self.num_inv_equ*output.size(1), 64, 4, 4)
        #elif 'Group' in self.config_name:
        #    output = output.view(output.size(0), output.size(1), 64, 4, 4)
        else:
            #output = output.view(output.size(0), output.size(1), 64, 4, 4)
            output = output.view(output.size(0), 64, 4, 4)

            #else:
            #    output = output.view(self.num_inv_equ * output.size(1), 64, 4, 4)

        #output = output.view(output.size(0), 64, 4, 4)
        #output = input
        if self.hidden_states:
            all_hidden_states = all_hidden_states + (output,)
        for i, hidden_layer in enumerate(self.hidden_layers):

            output = hidden_layer(output)
            if self.hidden_states:
                all_hidden_states = all_hidden_states + (output,)
        # output = torch.flatten(output, start_dim=1)
        output = self.active(output)
        #if torch.any(torch.isnan(output)):

        outputs = (output,) + (all_hidden_states,)
        return outputs

class CNNSmallNORBDecoder(CNN2DShapesDecoder):
    def __init__(self, config):
        super(CNNSmallNORBDecoder, self).__init__(config)

        modules = []
        self.dense1 = nn.Linear(self.latent_dim, config.dense_dim[1])
        self.dense2 = nn.Linear(config.dense_dim[1], 576)

        modules.append(CNNTrasnposedLayer(in_channels=64, out_channels=64))
        modules.append(CNNTrasnposedLayer(in_channels=64, out_channels=64))
        modules.append(CNNTrasnposedLayer(in_channels=64, out_channels=32))
        modules.append(CNNTrasnposedLayer(in_channels=32, out_channels=32))
        modules.append(nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1))
        self.hidden_layers = nn.ModuleList(modules)

    def forward(self, input):
        # input: [batch, latent_dim]
        all_hidden_states = ()
        output = self.dense1(input)
        output = self.relu(output)
        output = self.dense2(output)
        output = self.relu(output) # (B, ...)
        if 'CHIC' in self.config_name: #!= 'CNNCHICConfig':
            output = output.view(self.num_inv_equ*output.size(1), 64, 3, 3)

        else:
            #output = output.view(output.size(0), output.size(1), 64, 4, 4)
            output = output.view(output.size(0), 64, 3, 3)

        if self.hidden_states:
            all_hidden_states = all_hidden_states + (output,)
        for i, hidden_layer in enumerate(self.hidden_layers):

            output = hidden_layer(output)
            if self.hidden_states:
                all_hidden_states = all_hidden_states + (output,)

        output = self.active(output)


        outputs = (output,) + (all_hidden_states,)
        return outputs

class CNN2DShapesLieDecoder(CNN2DShapesDecoder):
    def __init__(self, config):
        super(CNN2DShapesLieDecoder, self).__init__(config)
        self.subgroup_sizes_ls = config.subgroup_sizes_ls
        self.subspace_sizes_ls = config.subspace_sizes_ls
        assert len(self.subgroup_sizes_ls) == len(self.subspace_sizes_ls)
        self.lie_alg_init_scale = 0.001
        self.no_exp = config.no_exp

        self.dense1 = nn.Linear(sum(self.subgroup_sizes_ls), config.dense_dim[0])
        self.dense2 = nn.Linear(config.dense_dim[0], 4*config.dense_dim[0])

        if self.no_exp:
            in_size = sum(self.subspace_sizes_ls)
            out_size = sum(self.subgroup_sizes_ls)
            self.fake_exp = nn.Sequential(
                nn.Linear(in_size, 4*in_size),
                nn.ReLU(True),
                nn.Linear(4*in_size, out_size)
            )
            for p in self.fake_exp.modules():
                if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear) or isinstance(p, nn.ConvTranspose2d):
                    torch.nn.init.xavier_uniform_(p.weight)
        self.lie_alg_basis_ls = nn.ParameterList([])
        for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls):
            mat_dim =int(math.sqrt(subgroup_size_i))
            assert mat_dim * mat_dim == subgroup_size_i
            for j in range(self.subspace_sizes_ls[i]):
                lie_alg_tmp, var_tmp =self.init_alg_basis(i, j, mat_dim, self.lie_alg_init_scale)
                self.lie_alg_basis_ls.append(lie_alg_tmp)

    def val_exp(self, x, lie_alg_basis_ls):
        lie_alg_basis_ls = [p * 1. for p in lie_alg_basis_ls
                            ]  # For torch.cat, convert param to tensor.
        lie_alg_basis = torch.cat(lie_alg_basis_ls,
                                  dim=0)[np.newaxis,
                                         ...]  # [1, lat_dim, mat_dim, mat_dim]
        lie_alg_mul = x[
            ..., np.newaxis, np.
            newaxis] * lie_alg_basis  # [b, lat_dim, mat_dim, mat_dim]
        lie_alg = torch.sum(lie_alg_mul, dim=1)  # [b, mat_dim, mat_dim]
        lie_group = torch.matrix_exp(lie_alg)  # [b, mat_dim, mat_dim]
        return lie_group

    def train_exp(self, x, lie_alg_basis_ls, mat_dim):
        lie_alg_basis_ls = [p * 1. for p in lie_alg_basis_ls
                            ]  # For torch.cat, convert param to tensor.
        lie_alg_basis = torch.cat(lie_alg_basis_ls,
                                  dim=0)[np.newaxis,
                                         ...]  # [1, lat_dim, mat_dim, mat_dim]
        lie_group = torch.eye(mat_dim, dtype=x.dtype).to(
            x.device)[np.newaxis, ...]  # [1, mat_dim, mat_dim]
        lie_alg = 0.
        latents_in_cut_ls = [x]
        for masked_latent in latents_in_cut_ls:
            lie_alg_sum_tmp = torch.sum(
                masked_latent[..., np.newaxis, np.newaxis] * lie_alg_basis,
                dim=1)
            lie_alg += lie_alg_sum_tmp  # [b, mat_dim, mat_dim]
            lie_group_tmp = torch.matrix_exp(lie_alg_sum_tmp)
            lie_group = torch.matmul(lie_group,
                                     lie_group_tmp)  # [b, mat_dim, mat_dim]
        return lie_group

    def init_alg_basis(self, i, j, mat_dim, lie_alg_init_scale):
        lie_alg_tmp = nn.Parameter(torch.normal(mean=torch.zeros(
            1, mat_dim, mat_dim),
            std=lie_alg_init_scale),
            requires_grad=True)
        var_tmp = nn.Parameter(
            torch.normal(torch.zeros(1, 1), lie_alg_init_scale))
        return lie_alg_tmp, var_tmp

    def forward(self, latents_in):
        latent_dim = list(latents_in.size())[-1]

        if self.no_exp:
            lie_group_tensor = self.fake_exp(latents_in)
        else:
            assert latent_dim == sum(self.subspace_sizes_ls)
            # Calc exp.
            lie_group_tensor_ls = []
            b_idx = 0
            for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls):
                mat_dim = int(math.sqrt(subgroup_size_i))
                e_idx = b_idx + self.subspace_sizes_ls[i]
                if self.subspace_sizes_ls[i] > 1:
                    if not self.training:
                        lie_subgroup = self.val_exp(
                            latents_in[:, b_idx:e_idx],
                            self.lie_alg_basis_ls[b_idx:e_idx])
                    else:
                        lie_subgroup = self.train_exp(
                            latents_in[:, b_idx:e_idx],
                            self.lie_alg_basis_ls[b_idx:e_idx], mat_dim)
                else:
                    lie_subgroup = self.val_exp(latents_in[:, b_idx:e_idx],
                                                self.lie_alg_basis_ls[b_idx:e_idx])
                lie_subgroup_tensor = lie_subgroup.view(-1, mat_dim * mat_dim)
                lie_group_tensor_ls.append(lie_subgroup_tensor)
                b_idx = e_idx
            lie_group_tensor = torch.cat(lie_group_tensor_ls,
                                         dim=1)  # [b, group_feat_size]

        output = self.dense1(lie_group_tensor)
        output = self.relu(output)
        output = self.dense2(output)
        output = self.relu(output)

        batch = output.size(0)
        output = output.view(batch, 64, 4, 4)
        for i, hidden_layer in enumerate(self.hidden_layers):
            output = hidden_layer(output)
        output = self.active(output)
        return (output, lie_group_tensor,) # after main decoder, after group decoder

    def gfeat(self, input):
        output = self.dense1(input)
        output = self.relu(output)
        output = self.dense2(output)
        output = self.relu(output)
        output = output.view(-1, 64, 4, 4)
        for i, hidden_layer in enumerate(self.hidden_layers):
            output = hidden_layer(output)
        output = self.active(output)
        return output

class CNNSmallNORBLieDecoder(CNNSmallNORBDecoder):
    def __init__(self, config):
        super(CNNSmallNORBLieDecoder, self).__init__(config)

        self.subgroup_sizes_ls = config.subgroup_sizes_ls
        self.subspace_sizes_ls = config.subspace_sizes_ls
        assert len(self.subgroup_sizes_ls) == len(self.subspace_sizes_ls)
        self.lie_alg_init_scale = 0.001
        self.no_exp = config.no_exp

        self.dense1 = nn.Linear(sum(self.subgroup_sizes_ls), config.dense_dim[0])
        self.dense2 = nn.Linear(config.dense_dim[0], 576)

        if self.no_exp:
            in_size = sum(self.subspace_sizes_ls)
            out_size = sum(self.subgroup_sizes_ls)
            self.fake_exp = nn.Sequential(
                nn.Linear(in_size, 4 * in_size),
                nn.ReLU(True),
                nn.Linear(4 * in_size, out_size)
            )
            for p in self.fake_exp.modules():
                if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear) or isinstance(p, nn.ConvTranspose2d):
                    torch.nn.init.xavier_uniform_(p.weight)
        self.lie_alg_basis_ls = nn.ParameterList([])
        for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls):
            mat_dim = int(math.sqrt(subgroup_size_i))
            assert mat_dim * mat_dim == subgroup_size_i
            for j in range(self.subspace_sizes_ls[i]):
                lie_alg_tmp, var_tmp = self.init_alg_basis(i, j, mat_dim, self.lie_alg_init_scale)
                self.lie_alg_basis_ls.append(lie_alg_tmp)

    def val_exp(self, x, lie_alg_basis_ls):
        lie_alg_basis_ls = [p * 1. for p in lie_alg_basis_ls
                            ]  # For torch.cat, convert param to tensor.
        lie_alg_basis = torch.cat(lie_alg_basis_ls,
                                  dim=0)[np.newaxis,
                                         ...]  # [1, lat_dim, mat_dim, mat_dim]
        lie_alg_mul = x[
                          ..., np.newaxis, np.
                              newaxis] * lie_alg_basis  # [b, lat_dim, mat_dim, mat_dim]
        lie_alg = torch.sum(lie_alg_mul, dim=1)  # [b, mat_dim, mat_dim]
        lie_group = torch.matrix_exp(lie_alg)  # [b, mat_dim, mat_dim]
        return lie_group

    def train_exp(self, x, lie_alg_basis_ls, mat_dim):
        lie_alg_basis_ls = [p * 1. for p in lie_alg_basis_ls
                            ]  # For torch.cat, convert param to tensor.
        lie_alg_basis = torch.cat(lie_alg_basis_ls,
                                  dim=0)[np.newaxis,
                                         ...]  # [1, lat_dim, mat_dim, mat_dim]
        lie_group = torch.eye(mat_dim, dtype=x.dtype).to(
            x.device)[np.newaxis, ...]  # [1, mat_dim, mat_dim]
        lie_alg = 0.
        latents_in_cut_ls = [x]
        for masked_latent in latents_in_cut_ls:
            lie_alg_sum_tmp = torch.sum(
                masked_latent[..., np.newaxis, np.newaxis] * lie_alg_basis,
                dim=1)
            lie_alg += lie_alg_sum_tmp  # [b, mat_dim, mat_dim]
            lie_group_tmp = torch.matrix_exp(lie_alg_sum_tmp)
            lie_group = torch.matmul(lie_group,
                                     lie_group_tmp)  # [b, mat_dim, mat_dim]
        return lie_group

    def init_alg_basis(self, i, j, mat_dim, lie_alg_init_scale):
        lie_alg_tmp = nn.Parameter(torch.normal(mean=torch.zeros(
            1, mat_dim, mat_dim),
            std=lie_alg_init_scale),
            requires_grad=True)
        var_tmp = nn.Parameter(
            torch.normal(torch.zeros(1, 1), lie_alg_init_scale))
        return lie_alg_tmp, var_tmp

    def forward(self, latents_in):
        latent_dim = list(latents_in.size())[-1]

        if self.no_exp:
            lie_group_tensor = self.fake_exp(latents_in)
        else:
            assert latent_dim == sum(self.subspace_sizes_ls)
            # Calc exp.
            lie_group_tensor_ls = []
            b_idx = 0
            for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls):
                mat_dim = int(math.sqrt(subgroup_size_i))
                e_idx = b_idx + self.subspace_sizes_ls[i]
                if self.subspace_sizes_ls[i] > 1:
                    if not self.training:
                        lie_subgroup = self.val_exp(
                            latents_in[:, b_idx:e_idx],
                            self.lie_alg_basis_ls[b_idx:e_idx])
                    else:
                        lie_subgroup = self.train_exp(
                            latents_in[:, b_idx:e_idx],
                            self.lie_alg_basis_ls[b_idx:e_idx], mat_dim)
                else:
                    lie_subgroup = self.val_exp(latents_in[:, b_idx:e_idx],
                                                self.lie_alg_basis_ls[b_idx:e_idx])
                lie_subgroup_tensor = lie_subgroup.view(-1, mat_dim * mat_dim)
                lie_group_tensor_ls.append(lie_subgroup_tensor)
                b_idx = e_idx
            lie_group_tensor = torch.cat(lie_group_tensor_ls,
                                         dim=1)  # [b, group_feat_size]

        output = self.dense1(lie_group_tensor)
        output = self.relu(output)
        output = self.dense2(output)
        output = self.relu(output)

        batch = output.size(0)
        output = output.view(batch, 64, 3, 3)
        for i, hidden_layer in enumerate(self.hidden_layers):
            output = hidden_layer(output)
        # output = self.hidden_layers(lie_group_tensor)
        # output = self.active(output)
        output = self.active(output)
        return (output, lie_group_tensor,)  # after main decoder, after group decoder

    def gfeat(self, input):
        output = self.dense1(input)
        output = self.relu(output)
        output = self.dense2(output)
        output = self.relu(output)
        output = output.view(-1, 64, 3, 3)
        for i, hidden_layer in enumerate(self.hidden_layers):
            output = hidden_layer(output)
        output = self.active(output)
        return output

class CNN3DShapesDecoder(CNN2DShapesDecoder):
    def __init__(self, config):
        super(CNN3DShapesDecoder, self).__init__(config)

        modules = []
        self.latent_dim = config.latent_dim
        self.hidden_states = config.hidden_states
        self.config_name = config.__class__.__name__
        self.num_inv_equ = config.num_inv_equ if 'CHIC' in self.config_name else None  # self.config_name =='CNNCHICConfig' else None
        # Design Decoder Factor-VAE ref
        self.dense1 = nn.Linear(self.latent_dim, config.dense_dim[1])
        self.dense2 = nn.Linear(config.dense_dim[1], 4 * config.dense_dim[0])
        self.relu = nn.ReLU(True)
        self.active = nn.Sigmoid()

        modules.append(CNNTrasnposedLayer(in_channels=64, out_channels=64))
        modules.append(CNNTrasnposedLayer(in_channels=64, out_channels=32))
        modules.append(CNNTrasnposedLayer(in_channels=32, out_channels=32))
        modules.append(nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1))
        self.hidden_layers = nn.ModuleList(modules)

class CNN3DShapesLieDecoder(CNN3DShapesDecoder):
    def __init__(self, config):
        super(CNN3DShapesLieDecoder, self).__init__(config)
        self.subgroup_sizes_ls = config.subgroup_sizes_ls
        self.subspace_sizes_ls = config.subspace_sizes_ls
        assert len(self.subgroup_sizes_ls) == len(self.subspace_sizes_ls)
        self.lie_alg_init_scale = 0.001
        self.no_exp = config.no_exp

        self.dense1 = nn.Linear(sum(self.subgroup_sizes_ls), config.dense_dim[0])
        self.dense2 = nn.Linear(config.dense_dim[0], 4*config.dense_dim[0])

        if self.no_exp:
            in_size = sum(self.subspace_sizes_ls)
            out_size = sum(self.subgroup_sizes_ls)
            self.fake_exp = nn.Sequential(
                nn.Linear(in_size, 4*in_size),
                nn.ReLU(True),
                nn.Linear(4*in_size, out_size)
            )
            for p in self.fake_exp.modules():
                if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear) or isinstance(p, nn.ConvTranspose2d):
                    torch.nn.init.xavier_uniform_(p.weight)
        self.lie_alg_basis_ls = nn.ParameterList([])
        for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls):
            mat_dim =int(math.sqrt(subgroup_size_i))
            assert mat_dim * mat_dim == subgroup_size_i
            for j in range(self.subspace_sizes_ls[i]):
                lie_alg_tmp, var_tmp =self.init_alg_basis(i, j, mat_dim, self.lie_alg_init_scale)
                self.lie_alg_basis_ls.append(lie_alg_tmp)

    def val_exp(self, x, lie_alg_basis_ls):
        lie_alg_basis_ls = [p * 1. for p in lie_alg_basis_ls
                            ]  # For torch.cat, convert param to tensor.
        lie_alg_basis = torch.cat(lie_alg_basis_ls,
                                  dim=0)[np.newaxis,
                                         ...]  # [1, lat_dim, mat_dim, mat_dim]
        lie_alg_mul = x[
            ..., np.newaxis, np.
            newaxis] * lie_alg_basis  # [b, lat_dim, mat_dim, mat_dim]
        lie_alg = torch.sum(lie_alg_mul, dim=1)  # [b, mat_dim, mat_dim]
        lie_group = torch.matrix_exp(lie_alg)  # [b, mat_dim, mat_dim]
        return lie_group

    def train_exp(self, x, lie_alg_basis_ls, mat_dim):
        lie_alg_basis_ls = [p * 1. for p in lie_alg_basis_ls
                            ]  # For torch.cat, convert param to tensor.
        lie_alg_basis = torch.cat(lie_alg_basis_ls,
                                  dim=0)[np.newaxis,
                                         ...]  # [1, lat_dim, mat_dim, mat_dim]
        lie_group = torch.eye(mat_dim, dtype=x.dtype).to(
            x.device)[np.newaxis, ...]  # [1, mat_dim, mat_dim]
        lie_alg = 0.
        latents_in_cut_ls = [x]
        for masked_latent in latents_in_cut_ls:
            lie_alg_sum_tmp = torch.sum(
                masked_latent[..., np.newaxis, np.newaxis] * lie_alg_basis,
                dim=1)
            lie_alg += lie_alg_sum_tmp  # [b, mat_dim, mat_dim]
            lie_group_tmp = torch.matrix_exp(lie_alg_sum_tmp)
            lie_group = torch.matmul(lie_group,
                                     lie_group_tmp)  # [b, mat_dim, mat_dim]
        return lie_group

    def init_alg_basis(self, i, j, mat_dim, lie_alg_init_scale):
        lie_alg_tmp = nn.Parameter(torch.normal(mean=torch.zeros(
            1, mat_dim, mat_dim),
            std=lie_alg_init_scale),
            requires_grad=True)
        var_tmp = nn.Parameter(
            torch.normal(torch.zeros(1, 1), lie_alg_init_scale))
        return lie_alg_tmp, var_tmp

    def forward(self, latents_in):
        latent_dim = list(latents_in.size())[-1]

        if self.no_exp:
            lie_group_tensor = self.fake_exp(latents_in)
        else:
            assert latent_dim == sum(self.subspace_sizes_ls)
            # Calc exp.
            lie_group_tensor_ls = []
            b_idx = 0
            for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls):
                mat_dim = int(math.sqrt(subgroup_size_i))
                e_idx = b_idx + self.subspace_sizes_ls[i]
                if self.subspace_sizes_ls[i] > 1:
                    if not self.training:
                        lie_subgroup = self.val_exp(
                            latents_in[:, b_idx:e_idx],
                            self.lie_alg_basis_ls[b_idx:e_idx])
                    else:
                        lie_subgroup = self.train_exp(
                            latents_in[:, b_idx:e_idx],
                            self.lie_alg_basis_ls[b_idx:e_idx], mat_dim)
                else:
                    lie_subgroup = self.val_exp(latents_in[:, b_idx:e_idx],
                                                self.lie_alg_basis_ls[b_idx:e_idx])
                lie_subgroup_tensor = lie_subgroup.view(-1, mat_dim * mat_dim)
                lie_group_tensor_ls.append(lie_subgroup_tensor)
                b_idx = e_idx
            lie_group_tensor = torch.cat(lie_group_tensor_ls,
                                         dim=1)  # [b, group_feat_size]

        output = self.dense1(lie_group_tensor)
        output = self.relu(output)
        output = self.dense2(output)
        output = self.relu(output)

        batch = output.size(0)
        output = output.view(batch, 64, 4, 4)
        for i, hidden_layer in enumerate(self.hidden_layers):
            output = hidden_layer(output)
        #output = self.hidden_layers(lie_group_tensor)
        #output = self.active(output)
        output = self.active(output)
        return (output, lie_group_tensor,) # after main decoder, after group decoder

    def gfeat(self, input):
        output = self.dense1(input)
        output = self.relu(output)
        output = self.dense2(output)
        output = self.relu(output)
        output = output.view(-1, 64, 4, 4)
        for i, hidden_layer in enumerate(self.hidden_layers):
            output = hidden_layer(output)
        output = self.active(output)
        return output
