import os
import torch
import shutil
import logging
from src.seed import set_seed
from src.files import make_run_files
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from torch.optim import SGD, Adam
OPTIMIZER = {
    'sgd': SGD,
    'adam': Adam,
}

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def eval(train_dataset, model, loss_fn, args):
    set_seed(args)
    # set for tensorboard dir
    save_files = make_run_files(args)
    run_file = os.path.join(args.run_file, args.model_type, save_files)

    if args.write:
        if os.path.exists(run_file):
            shutil.rmtree(run_file)
            os.makedirs(run_file)
        else:
            os.makedirs(run_file)
        # tb_writer = SummaryWriter(run_file)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.test_batch_size,
                                  drop_last=False, pin_memory=True)
    global_step = 0
    learning_rate = args.lr_rate
    t_total = len(train_dataloader)

    #multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train
    logger.info("***********Running Model Evaluation***********")
    logger.info(" Num examples = %d", len(train_dataset))
    logger.info(" Num Epochs = %d", args.num_epoch)
    logger.info(" Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        " Total train batch size = %d",
        args.train_batch_size * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("Total optimization steps = %d", t_total)

    # Common loss function
    tr_elbo, logging_elbo = 0.0, 0.0
    tr_reconst_err, logging_reconst_err = 0.0, 0.0
    tr_kld_err, logging_kld_err = 0.0, 0.0
    tr_total_loss, logging_total_loss = 0.0, 0.0

    # for beta-TCVAE
    tr_mi, logging_mi = 0.0, 0.0
    tr_tc, logging_tc = 0.0, 0.0

    # for Info VAE
    tr_mmd, logging_mmd = 0.0, 0.0

    # for GROUP-VAEs
    tr_commutative, logging_commutative = 0.0, 0.0
    tr_contrastive, logging_contrastive = 0.0, 0.0
    tr_equivariant, logging_equivariant = 0.0, 0.0
    # for Control VAE
    tr_beta, logging_beta = 0.0, 0.0

    # for Commutative VAE
    tr_group, logging_group = 0.0, 0.0

    # for common
    total_loss = None
    kld = None
    # for beta-TCVAE
    mi = None
    tc = None
    # for Info VAE
    mmd = None
    # for GORUP-VAEs
    commutative = None
    contrastive = None
    equivariant = None
    mse = None
    positive = None
    #negative= None

    # for Control VAE
    beta = None
    # for Commutative VAE
    group = None

    set_seed(args)

    iteration_per_epoch = len(train_dataloader)

    results = {}
    #model.zero_grad()
    iteration = tqdm(train_dataloader, desc="Iteration")

    for i, (data, class_label) in enumerate(iteration):
        with torch.no_grad():
            model.eval()
            data = data.to(device)
            outputs = model(data, loss_fn)

            reconst_err, kld_err = outputs[0]['obj']['reconst'], outputs[0]['obj']['kld']

            if args.model_type == 'betatcvae' or args.model_type == 'groupbetatcvae':
                mi = outputs[0]['obj']['mi']
                tc = outputs[0]['obj']['tc']
            elif args.model_type == 'commutativevae' or args.model_type == 'groupcommutativevae':
                group = outputs[0]['obj']['group']

            if args.model_type == 'controlvae':
                total_loss = reconst_err + kld_err
            elif args.model_type == 'betavae':
                total_loss = reconst_err + args.beta * kld_err
            elif args.model_type == 'betatcvae':
                total_loss = reconst_err + \
                             args.alpha * mi + \
                             args.beta * tc + \
                             args.gamma * kld_err
            elif args.model_type == 'commutativevae':
                total_loss = reconst_err + kld_err + group
            elif args.model_type == 'groupvae':
                total_loss = reconst_err + args.beta * kld_err
            elif args.model_type == 'groupbetatcvae':
                total_loss = reconst_err + \
                             mi + args.beta * tc + kld_err

            elif args.model_type == 'groupcommutativevae':
                total_loss = reconst_err + kld_err + \
                             group

            elif args.model_type == 'groupcontrolvae':
                total_loss = reconst_err + kld_err

            if args.n_gpu > 1:
                total_loss = total_loss.mean()
                reconst_err = reconst_err.mean()
                kld_err = kld_err.mean()

                if args.model_type == 'betatcvae' or args.model_type == 'groupbetatcvae':
                    mi = mi.mean()
                    tc = tc.mean()
                elif args.model_type == 'commutativevae' or args.model_type == 'groupcommutativevae':
                    group = group.mean()

            elbo = -(reconst_err + kld_err)
            tr_total_loss += total_loss.item()
            tr_elbo += elbo.item()
            tr_reconst_err += reconst_err.item()
            tr_kld_err += kld_err.item()

            if args.model_type == 'betatcvae' or args.model_type == 'groupbetatcvae':
                tr_mi += mi.item()
                tr_tc += tc.item()
            elif args.model_type == 'commutativevae' or args.model_type == 'groupcommutativevae':
                tr_group = group.item()
            global_step += 1

    results['elbo'] = tr_elbo / t_total
    results['reconst'] = tr_reconst_err / t_total
    results['kld'] = tr_kld_err / t_total

    if args.model_type == 'betatcvae' or args.model_type == 'groupbetatcvae':
        results['mi'] = tr_mi / t_total
        results['tc'] = tr_tc / t_total
    elif args.model_type == 'commutativevae' or args.model_type == 'groupcommutativevae':
        results['group'] = tr_group / t_total

    return results