import torch
import torch.nn as nn
import math
from src.constants import DATA_CLASSES
import pdb

class CNNControlVAE(nn.Module):
    def __init__(self,config):
        super(CNNControlVAE, self).__init__()
        self.t_total = config.t_total
        self.init_kld = 0.5
        self.const_kld = config.const_kld
        self.min_beta = config.min_beta
        self.max_beta = config.max_beta
        self.k_i, self.k_p = config.k_i, config.k_p
        self.init_beta = 0.0
        self.init_I = 0.0

        encoder, decoder = DATA_CLASSES[config.dataset]
        self.encoder = encoder(config)
        self.decoder = decoder(config)

    def forward(self, input, loss_fn):
        encoder_output = self.encoder(input)
        decoder_output = self.decoder(encoder_output[0])
        outputs = (encoder_output,) + (decoder_output,)# ((z,mu, logvar,(encoder)), (reconst, (decoder)))
        loss = self.loss(input, outputs, loss_fn)
        loss = (loss,) + (encoder_output,) + (decoder_output,)
        # ((elbo, reconst_err, kld_err, id_mea, id_var), (z,mu, logvar,(encoder)), (reconst, (decoder)))
        return loss

    # Add loss function
    def loss(self, input, outputs, loss_fn):
        result = {'elbo': {}, 'obj': {}, 'id': {}}
        batch = input.size(0)
        reconsted_images = outputs[1][0]
        z, mu, logvar = outputs[0][0].squeeze(), outputs[0][1].squeeze(), outputs[0][2].squeeze()

        #criteria = nn.BCELoss(reduction='sum')
        reconst_err = loss_fn(reconsted_images, input) / batch
        kld_err = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=-1))
        beta = self.pi_ctrl(kld_err, self.t_total)
        result['obj']['reconst'] = reconst_err
        result['obj']['kld'] = beta * kld_err if self.training else kld_err
        result['obj']['beta'] = beta
        return result

    #ControlVAE paper algorithm
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

    def log_density_gaussian(self, x, mu, logvar):
        #f(x) = \frac{1}{sigma * sqrt(2 * pi)} exp(-0.5 * ((z - mu) / sigma) ** 2): Gaussian Distribution
        norm = -0.5 * (math.log(2 * math.pi) + logvar)
        log_density = norm - 0.5 * (x - mu) ** 2 * torch.exp(-logvar)
        return log_density

    def init_weights(self):
        for n, p in self.named_parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            #else:
                #nn.init.zeros_(p.data)
