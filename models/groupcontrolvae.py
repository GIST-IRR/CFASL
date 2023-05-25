import torch
import torch.nn as nn
from models.group_action_layer import Group_Elements
from src.constants import DATA_CLASSES

class CNNGroupControlVAE(nn.Module):

    def __init__(self, config):
        super(CNNGroupControlVAE, self).__init__()
        encoder, decoder = DATA_CLASSES[config.dataset]
        self.encoder = encoder(config)
        self.decoder = decoder(config)
        self.group_action_layer = Group_Elements(config)

        self.lamb = config.lamb

        self.t_total = config.t_total
        self.init_kld = 0.5
        self.const_kld = config.const_kld
        self.min_beta = config.min_beta
        self.max_beta = config.max_beta
        self.k_i, self.k_p = config.k_i, config.k_p
        self.init_beta = 0.0
        self.init_I = 0.0

    def forward(self, input, loss_fn):
        #if self.training:
        encoder_output = self.encoder(input) # (z, mu, logvar, (option: all_hidden_representation))
        batch, dim = encoder_output[0].size()
        slice = batch//2

        #transformed_z1, transformed_z2 = encoder_output[0][:batch/2], encoder_output[0][batch/2:]
        transformed_z1, transformed_z2, group_action, equivariant, attn, orthogonal_loss, parallel_loss, commut_loss, sparse_loss, sector_loss, sub_symmetries = self.group_action_layer(
            encoder_output[1], encoder_output[2], encoder_output[0])  # commutative_loss

        transformed_z = torch.cat([transformed_z2, transformed_z1], dim=0) # [batch, dim]
        merge_z = torch.cat([encoder_output[0], transformed_z], dim=0) # [2*batch, dim])

        decoder_output = self.decoder(merge_z)
        outputs = (encoder_output,) + (decoder_output,)

        loss = self.loss(input, outputs, loss_fn)
        loss['obj']['orthogonal'] = orthogonal_loss
        loss['obj']['parallel'] = parallel_loss
        loss['obj']['commutative'] = commut_loss
        loss['obj']['sector'] = sector_loss
        loss['obj']['sparse'] = sparse_loss
        # encoder equivariant loss
        loss['obj']['equivariant_1'] = equivariant
        # decoder equivariant loss
        loss['obj']['equivariant_2'] = loss_fn(outputs[1][0][batch:], input) / batch
        loss = (loss,) + (encoder_output,) + (decoder_output,) + (group_action,) + (attn,) + (sub_symmetries,)
        return loss

    def loss(self, input, outputs, loss_fn):
        result = {'elbo': {}, 'obj': {}, 'id': {}}
        batch = input.size(0)
        reconsted_images = outputs[1][0][:batch]  # [batch, c, h, w]
        z, mu, logvar = outputs[0][0].squeeze(), outputs[0][1].squeeze(), outputs[0][2].squeeze()

        reconst_err = loss_fn(reconsted_images, input) / batch
        kld_err = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=-1))
        beta = self.pi_ctrl(kld_err, self.t_total)

        result['obj']['reconst'] = reconst_err.unsqueeze(0)
        result['obj']['kld'] = beta * kld_err.unsqueeze(0) if self.training else kld_err.unsqueeze(0)
        result['obj']['beta'] = beta

        return result

    def pi_ctrl(self, kld, t_total):

        err = self.init_kld - kld # alg line 6
        p = self.k_p / (1 + torch.exp(err))  # alg line 7

        if self.init_beta >= self.min_beta and self.init_beta <= self.max_beta: # alg line 8
            self.init_I = self.init_I - self.k_i * err # alg line 9
        beta = p + self.init_I + self.min_beta # alg line 13

        if beta > self.max_beta: # alg line 14
            beta = self.max_beta # alg line 15
        if beta < self.min_beta: # alg line 17
            beta = self.min_beta # alg line 18

        self.init_kld = self.init_kld + (self.const_kld - 0.5) / t_total
        return beta

    def init_weights(self):
        for n, p in self.named_parameters():
            if p.data.ndimension() >= 2 and 'group_elements' in n:
                nn.init.trunc_normal_(p.data)
            elif p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.uniform_(p.data)
