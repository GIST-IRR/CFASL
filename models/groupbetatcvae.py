import torch
import torch.nn as nn
import math
from models.group_action_layer import Group_Elements
from src.constants import DATA_CLASSES


class CNNGroupBetaTCVAE(nn.Module):
    def __init__(self, config):
        super(CNNGroupBetaTCVAE, self).__init__()

        encoder, decoder = DATA_CLASSES[config.dataset]
        self.encoder = encoder(config)
        self.decoder = decoder(config)
        self.dataset_size = config.dataset_size
        self.group_action_layer = Group_Elements(config)

    def forward(self, input, loss_fn):
        encoder_output = self.encoder(input)
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

        decoder_output = self.decoder(merge_z)
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
        loss["obj"]["equivariant_2"] = loss_fn(outputs[1][0][batch:], input) / batch
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
        reconsted_images = outputs[1][0][:batch]  # [batch, c, h, w]
        z, mu, logvar = (
            outputs[0][0].squeeze(),
            outputs[0][1].squeeze(),
            outputs[0][2].squeeze(),
        )

        zeros = torch.zeros_like(z)
        logqzx = self.log_density_gaussian(z, mu, logvar).sum(dim=1)
        logpz = self.log_density_gaussian(z, zeros, zeros).sum(dim=1)  # size: batch
        _logqz = self.log_density_gaussian(
            z.view(batch, 1, -1), mu.view(1, batch, -1), logvar.view(1, batch, -1)
        )  # size: (batch, batch, dim)
        # Minibatch Stratified Sampling
        # Reference
        # Isolating Sources of Disentanglement in VAEs.
        stratified_weight = (self.dataset_size - batch + 1) / (
            self.dataset_size * (batch - 1)
        )
        importance_weights = (
            torch.Tensor(batch, batch).fill_(1 / (batch - 1)).to(z.device)
        )
        importance_weights.view(-1)[::batch] = 1 / self.dataset_size
        importance_weights.view(-1)[1::batch] = stratified_weight
        importance_weights[batch - 2, 0] = stratified_weight
        log_importance_weights = importance_weights.log()
        _logqz += log_importance_weights.view(batch, batch, 1)

        logqz_prod = torch.logsumexp(_logqz, dim=1, keepdim=False).sum(
            1
        )  # - math.log(batch)).sum(1)  # size: batch
        logqz = torch.logsumexp(
            _logqz.sum(2), dim=1, keepdim=False
        )  # - math.log(batch)

        reconst_err = loss_fn(reconsted_images, input) / batch  # * input.size(-1) ** 2
        mi = torch.mean(logqzx - logqz)
        tc_err = torch.mean(logqz - logqz_prod)
        kld_err = torch.mean(logqz_prod - logpz)

        result["obj"]["reconst"] = reconst_err.unsqueeze(0)
        result["obj"]["kld"] = kld_err.unsqueeze(0)
        result["obj"]["mi"] = mi.unsqueeze(0)
        result["obj"]["tc"] = tc_err.unsqueeze(0)
        return result

    def log_density_gaussian(self, x, mu, logvar):
        # f(x) = \frac{1}{sigma * sqrt(2 * pi)} exp(-0.5 * ((z - mu) / sigma) ** 2): Gaussian Distribution
        norm = -0.5 * (math.log(2 * math.pi) + logvar)
        log_density = norm - 0.5 * (x - mu) ** 2 * torch.exp(-logvar)
        return log_density

    def init_weights(self):
        for n, p in self.named_parameters():
            if p.data.ndimension() >= 2 and "group_elements" in n:
                nn.init.trunc_normal_(p.data)
            elif p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.uniform_(p.data)
