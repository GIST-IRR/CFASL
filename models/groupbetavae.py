import torch
import torch.nn as nn
from models.group_action_layer import Group_Elements
from src.constants import DATA_CLASSES

class CNNGroupVAE(nn.Module):

    def __init__(self, config):
        super(CNNGroupVAE, self).__init__()
        encoder, decoder = DATA_CLASSES[config.dataset]
        self.encoder = encoder(config)
        self.decoder = decoder(config)
        self.group_action_layer = Group_Elements(config)

    def forward(self, input, loss_fn):
        encoder_output = self.encoder(input) # (z, mu, logvar, (option: all_hidden_representation))
        batch, dim = encoder_output[0].size()
        slice = batch // 2

        transformed_z1, transformed_z2, group_action, equivariant, attn, orthogonal_loss, parallel_loss, commut_loss, sparse_loss, sector_loss, sub_symmetries = self.group_action_layer(encoder_output[1],encoder_output[2],encoder_output[0]) #commutative_loss


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

        result['obj']['reconst'] = reconst_err.unsqueeze(0)
        result['obj']['kld'] = kld_err.unsqueeze(0)
        return result

    def init_weights(self):
        for n, p in self.named_parameters():
            if p.data.ndimension() >= 2 and 'group_elements' in n:
                nn.init.trunc_normal_(p.data)
            elif p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.uniform_(p.data)
