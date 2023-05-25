import torch
import logging
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from torch.optim import SGD, Adam
OPTIMIZER = {
    'sgd': SGD,
    'adam': Adam,
}

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

############################################################
# Qualitative Analysis
############################################################
def qualitative(dataset, model, args):
    logger.info("***********Qualitative Analysis***********")
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.quali_sampling)
    iteration = tqdm(train_dataloader, desc='Iteration')

    for k, (data, class_label) in enumerate(iteration):
        with torch.no_grad():
            model.eval()
            new_zs = []
            data = data.to(device)
            #dataset = dataset.to(device)
            outputs = model.encoder(data)

            z = outputs[0] # (Batch, dimension)
            mean = outputs[1]
            logvar = outputs[2]
            z = changed_latent_vector_value(mean, logvar, z, interval=args.interval)

            outputs = model.decoder(z)
            reconst = outputs[0] #outputs[2][0] if "commutative" not in args.model_type else outputs[2] #reconstruction imgs
            new_outputs = []
            for i in range(data.size(0)):
                new_outputs.append(data[i].unsqueeze(0))
                new_outputs.append(reconst[i*args.interval: (i+1)*args.interval, :, :, :])
            new_outputs = torch.cat(new_outputs, dim=0)

            if k == 0:
                break
    return new_outputs

def changed_latent_vector_value(mean, logvar, latent_vector, interval):
    dim = latent_vector.size(0)  # Batch == dimension

    mask = torch.ones_like(latent_vector) - torch.eye(dim).to(device)  # diagonal is zeros and others are ones

    # set latent z values from -2 to 2
    latent_vector = latent_vector * mask
    interval = torch.arange(-2, 2.1, 4/(interval-1)).to(device) # [latent_dim]
    interval = interval.unsqueeze(-1) # [latent_dim, 1]
    interval = interval.unsqueeze(2).expand(*interval.size(), interval.size(1)) # [latent_dim, 1, 1]
    interval = interval * torch.eye(dim).to(device) # extend to 3-D tensor # [latent_dim, latent_dim, latent_dim]

    latent_vector = (latent_vector + interval).permute(1, 0, 2).reshape(-1, dim) #(Batch * interval , dim)
    return latent_vector

