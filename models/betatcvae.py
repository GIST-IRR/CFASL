import torch
import torch.nn as nn
import math
from src.constants import DATA_CLASSES


# CNN VAE
class CNNBetaTCVAE(nn.Module):
    def __init__(self, config):
        super(CNNBetaTCVAE, self).__init__()
        encoder, decoder = DATA_CLASSES[config.dataset]
        self.encoder = encoder(config)
        self.decoder = decoder(config)
        self.dataset_size = config.dataset_size  # coding할것.

    def forward(self, input, loss_fn):
        encoder_output = self.encoder(input)
        decoder_output = self.decoder(encoder_output[0])
        outputs = (encoder_output,) + (
            decoder_output,
        )  # ((z,mu, logvar,(encoder)), (reconst, (decoder)))
        loss = self.loss(input, outputs, loss_fn)
        loss = (loss,) + (encoder_output,) + (decoder_output,)
        # ((elbo, reconst_err, kld_err, id_mea, id_var), (z,mu, logvar,(encoder)), (reconst, (decoder)))
        return loss

    # Add loss function
    def loss(self, input, outputs, loss_fn):
        result = {"elbo": {}, "obj": {}, "id": {}}
        batch = input.size(0)
        reconsted_images = outputs[1][0]
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

        # pdb.set_trace()
        logqz_prod = torch.logsumexp(_logqz, dim=1, keepdim=False).sum(
            1
        )  # - math.log(batch)).sum(1)  # size: batch
        logqz = torch.logsumexp(
            _logqz.sum(2), dim=1, keepdim=False
        )  # - math.log(batch)

        # criteria = nn.BCELoss(reduction='sum')
        reconst_err = loss_fn(reconsted_images, input) / batch  # * input.size(-1) ** 2
        # kld_err = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=-1))
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
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
