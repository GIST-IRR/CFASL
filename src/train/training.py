# for model training (common settings)
import pdb
import pickle
import numpy as np
# import wandb
import os
import torch
import shutil
import logging
import torch.nn.functional as F
from src.seed import set_seed
from src.files import make_run_files
# from src.Constants import OPTIMIZER
from src.optimizer import get_constant_schedule, get_linear_schedule_with_warmup

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
#from src.analysis.grid_ploting import grid_ploting

from torch.optim import SGD, Adam
OPTIMIZER = {
    'sgd': SGD,
    'adam': Adam,
}

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def factor_train():
    return

def train(train_dataset, num_epochs, model, loss_fn, args):
    optimizer = None
    set_seed(args)
    # set for tensorboard dir
    save_files = make_run_files(args)
    run_file = os.path.join(args.run_file, args.model_type, save_files)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  drop_last=True, pin_memory=True)
    global_step = 0
    learning_rate = args.lr_rate
    t_total = len(train_dataloader) * args.num_epoch if args.max_steps == 0 else args.max_steps
    args.t_total = t_total

    if args.optimizer == 'adam':
        main_list, sub_list = [], []
        for n, p in model.named_parameters():
            if 'linear' in n:
                sub_list.append(p)
            else:
                main_list.append(p)
        optimizer = OPTIMIZER[args.optimizer]([{'params': main_list},
                                               {'params': sub_list, 'lr':2e-5}],
                                              lr=learning_rate,
                                              betas=(0.9, 0.999),
                                              weight_decay=args.weight_decay)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total) if args.scheduler == 'linear' else get_constant_schedule(optimizer)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.output_dir, args.model_type, save_files, "optimizer.pt")) and \
            os.path.isfile(os.path.join(args.output_dir, args.model_type, save_files, "scheduler.pt")):
        optimizer.load_state_dict(
            torch.load(os.path.join(args.output_dir, args.model_type, save_files, "optimizer.pt")))
        scheduler.load_state_dict(
            torch.load(os.path.join(args.output_dir, args.model_type, save_files, "scheduler.pt")))



    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        student, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train
    logger.info("***********Running Model Training***********")
    logger.info(" Num examples = %d", len(train_dataset))
    logger.info(" Num Epochs = %d", args.num_epoch)
    logger.info(" Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        " Total train batch size = %d",
        args.train_batch_size * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info( "Total optimization steps = %d", t_total)

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

    # for Group-VAEs
    tr_orth, logging_orth = 0.0, 0.0
    tr_parl, logging_parl = 0.0, 0.0
    tr_sec, logging_sec = 0.0, 0.0
    tr_equivariant_2, logging_equivariant_2 = 0.0, 0.0
    tr_commutative, logging_commutative = 0.0, 0.0
    tr_equivariant, logging_equivariant = 0.0, 0.0
    tr_sparse, logging_sparse = 0.0, 0.0
    tr_symmetric, logging_symmetric = 0.0, 0.0
    # To check difference between z1 and z2
    tr_difference, logging_difference = 0.0, 0.0
    # To check difference between symmetry and Identity matrix
    tr_diffsym, logging_diffsym = 0.0, 0.0
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
    # for Group-VAEs
    orth = None
    parl = None
    sec = None
    equivariant_2 = None
    commutative = None
    equivariant = None
    sparse = None
    symmetric = None
    # To check difference between z1 and z2
    difference = None
    # To check difference between symmetry and Identity matrix
    diffsym = None
    # for Control VAE
    beta = None
    # for Commutative VAE
    group = None

    set_seed(args)
    attn_distribution, symmetries = None, None
    iteration_per_epoch = len(train_dataloader)

    model.zero_grad()

    # wandb.init(
    #     project=args.project_name,
    #     name=run_file
    # )
    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        iteration = tqdm(train_dataloader, desc="Iteration")

        for i, (data, class_label) in enumerate(iteration):
            batch = data.size(0)
            model.train()
            data = data.to(device)
            outputs = model(data, loss_fn)

            reconst_err, kld_err = outputs[0]['obj']['reconst'], outputs[0]['obj']['kld']

            if 'group' in args.model_type:
                orth = outputs[0]['obj']['orthogonal']
                parl = outputs[0]['obj']['parallel']
                sec = outputs[0]['obj']['sector']
                equivariant_2 = outputs[0]['obj']['equivariant_2']
                commutative = outputs[0]['obj']['commutative']
                equivariant = outputs[0]['obj']['equivariant_1']
                sparse = outputs[0]['obj']['sparse']
                z1, z2 = outputs[1][0][:batch//2].detach(), outputs[1][0][batch//2:].detach()
                difference = F.mse_loss(z1, z2)
                diffsym = torch.sum(torch.abs(outputs[3] - torch.eye(args.latent_dim).to(device)))
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
                total_loss = reconst_err + args.alpha * (orth+parl+commutative+sparse) + args.beta * kld_err + \
                             args.gamma * sec + args.lamb * equivariant + args.epsilon * equivariant_2
            elif args.model_type == 'groupbetatcvae':
                total_loss = reconst_err + \
                             mi + args.alpha * (orth+parl+commutative+sparse) + args.beta * tc + kld_err + \
                             args.gamma * sec + args.lamb * equivariant + args.epsilon * equivariant_2

            elif args.model_type == 'groupcommutativevae':
                total_loss = reconst_err + kld_err + \
                             group + args.alpha * (orth+parl+commutative+sparse) + \
                             args.gamma * sec + args.lamb * equivariant + args.epsilon * equivariant_2

            elif args.model_type == 'groupcontrolvae':
                total_loss = reconst_err + kld_err + args.alpha * (orth+parl+commutative+sparse) + \
                             args.gamma * sec + args.lamb * equivariant + args.epsilon * equivariant_2

            if args.n_gpu > 1:
                total_loss = total_loss.mean()
                reconst_err = reconst_err.mean()
                kld_err = kld_err.mean()

                if 'group' in args.model_type:
                    orth = orth.mean()
                    parl = parl.mean()
                    sec = sec.mean()
                    equivariant_2 = equivariant_2.mean()
                    commutative = commutative.mean()
                    equivariant = equivariant.mean()
                    difference = difference.mean()
                    sparse = sparse.mean()

                if args.model_type == 'betatcvae' or 'groupbetatcvae':
                    mi = mi.mean()
                    tc = tc.mean()
                elif args.model_type == 'commutativevae' or 'groupcommutativevae':
                    group = group.mean()

            elbo = -(reconst_err + kld_err)
            tr_total_loss += total_loss.item()
            tr_elbo += elbo.item()
            tr_reconst_err += reconst_err.item()
            tr_kld_err += kld_err.item()
            if "group" in args.model_type:
                tr_orth += orth.item()
                tr_parl += parl.item()
                tr_sec += sec.item()
                tr_equivariant_2 += equivariant_2.item()
                tr_commutative += commutative.item()
                tr_equivariant += equivariant.item()
                tr_difference += difference.item()
                tr_diffsym += diffsym.item()
                tr_sparse += sparse.item()

            if args.model_type == 'betatcvae' or args.model_type == 'groupbetatcvae':
                tr_mi += mi.item()
                tr_tc += tc.item()
            elif args.model_type == 'commutativevae' or args.model_type == 'groupcommutativevae':
                tr_group = group.item()

            if args.fp16:
                with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                total_loss.backward()
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            # Write wandb

            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % (args.logging_steps) == 0:
                logs = {}
                logs["00.ELBO"] = (tr_elbo - logging_elbo) / args.logging_steps
                logs["01.Total_Loss"] = (tr_total_loss - logging_total_loss) / args.logging_steps
                logs["02.Reconstruction_Loss"] = (tr_reconst_err - logging_reconst_err) / args.logging_steps
                logs["03.Kullback-Reibler_Loss"] = (tr_kld_err - logging_kld_err) / args.logging_steps

                if 'group' in args.model_type:
                    logs["Sector_loss"] = (tr_sec - logging_sec) / args.logging_steps
                    logs["GROUP_Orthogonal"] = (tr_orth - logging_orth) / args.logging_steps
                    logs["GROUP_Parallel"] = (tr_parl - logging_parl) / args.logging_steps
                    logs["GROUP_equivariant_2"] = (tr_equivariant_2 - logging_equivariant_2) / args.logging_steps
                    logs["GROUP_commutatve"] = (tr_commutative - logging_commutative) / args.logging_steps
                    logs["GROUP_equivariant"] = (tr_equivariant - logging_equivariant) / args.logging_steps
                    logs["Z_difference"] = (tr_difference - logging_difference) / args.logging_steps
                    logs["Sym_difference"] = (tr_diffsym - logging_diffsym) / args.logging_steps
                    logs["Sparsity"] = (tr_sparse - logging_sparse) / args.logging_steps

                    logging_orth = tr_orth
                    logging_parl = tr_parl
                    logging_sec = tr_sec
                    logging_equivariant_2 = tr_equivariant_2
                    logging_commutative = tr_commutative
                    logging_equivariant = tr_equivariant
                    logging_difference = tr_difference
                    logging_diffsym = tr_diffsym
                    logging_sparse = tr_sparse

                if args.model_type == 'betatcvae' or args.model_type == 'groupbetatcvae':
                    logs["TC_mutual_information"] = (tr_mi - logging_mi) / args.logging_steps
                    logs["TC_total_correlation"] = (tr_tc - logging_tc) / args.logging_steps
                    logging_mi = tr_mi
                    logging_tc = tr_tc
                elif args.model_type == 'commutativevae' or args.model_type == 'groupcommutativevae':
                    logs["Commutative_group"] = (tr_group - logging_group) / args.logging_steps
                    logging_group = tr_group

                logging_elbo = tr_elbo
                logging_total_loss = tr_total_loss
                logging_reconst_err = tr_reconst_err
                logging_kld_err = tr_kld_err

                learning_rate_scalar = scheduler.get_last_lr()[0]
                logs["learning_rate"] = learning_rate_scalar

            # Write model parameters (checkpoint)
            if (args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0) or \
                    global_step == args.max_steps or global_step == iteration_per_epoch * args.num_epoch: # save in last step
                output_dir = os.path.join(args.output_dir, args.model_type, save_files, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (model.module if hasattr(model, "module") else model)
                torch.save(model_to_save.state_dict(), os.path.join(output_dir, "model.pt"))
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

                # save attention distribution and symmetries (each checkpoint)
                with open(os.path.join(output_dir, 'attn_distribution.pickle'), 'wb') as f:
                    pickle.dump(attn_distribution, f)
                with open(os.path.join(output_dir, 'symmetries.pickle'), 'wb') as f:
                    pickle.dump(symmetries, f)

            if args.max_steps > 0 and global_step >= args.max_steps:
                iteration.close()
                return

    return







