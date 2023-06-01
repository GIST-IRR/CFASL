import pdb
import os
import torch
import torch.nn as nn
import logging
import argparse
import csv

# from src.Constants import DATA_CLASSES
from src.constants import DATA_HIDDEN_DIM
from src.constants import DATA_STEPS
from src.constants import DATALOADER

# from src.Constants import MODEL_CLASSES
# from src.Constants import OPTIMIZER
from configs.utils import (
    CNNVanillaVAEConfig,
    CNNBetaVAEConfig,
    CNNBetaTCVAEConfig,
    ControlVAEConfig,
    CNNLieConfig,
)
from configs.config import (
    GroupActionVAEConfig,
    GroupActionBetaTCVAEConfig,
    GroupActionControlVAEConfig,
    GroupActionCommutativeVAEConfig,
)


from models.betavae import CNNBetaVAE
from models.betatcvae import CNNBetaTCVAE
from models.commutativevae import CommutativeVAE
from models.controlvae import CNNControlVAE

from models.groupbetavae import CNNGroupVAE
from models.groupbetatcvae import CNNGroupBetaTCVAE
from models.groupcontrolvae import CNNGroupControlVAE
from models.groupcommutativevae import CNNGroupCommutativeVAE


from src.info import write_info
from src.files import make_run_files
from src.utils import load_model

from src.train.training import train
from src.train.evaluation import eval
from src.disent_metrics.eval import estimate_all_distenglement

from src.analysis_tools.common_quali import qualitative
from src.analysis_tools.symmetries import extract_group_actions
from src.analysis_tools.largest_kld import comparing_baseline
from src.analysis_tools.plots import plotting_3d
from src.analysis_tools.eigen import analysis_of_basis

from torch.utils.data import DataLoader
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    # set device info
    parser.add_argument(
        "--device_idx",
        type=str,
        default="cuda:0",
        required=True,
        help="set GPU index, i.e. cuda:0,1,2 ...",
    )
    parser.add_argument(
        "--n_gpu",
        type=int,
        default=0,
        required=False,
        help="number of available gpu",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    # DATASETS
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/s1_u1/datasets/disentanglement/2D_shapes/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
        required=False,
        help="dataset directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["dsprites", "shapes3d", "car", "smallnorb", "celeba", "cdsprites"],
        required=True,
        help="Choose Dataset",
    )

    # model save directory
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/s1_u1/checkpoints/VAEs",
        required=True,
        help="model saving directory",
    )
    parser.add_argument(
        "--run_file",
        type=str,
        default="/home/s1_u1/runs/VAEs/pilot",
        required=False,
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="/home/s1_u1/checkpoints/VAEs",
        required=False,
        help="model performance saving directory",
    )
    # Only for evaluation
    parser.add_argument(
        "--model_dir",
        type=str,
        default="model_dir",
        required=False,
        help="trained model directory",
    )
    parser.add_argument(
        "--steps",
        type=int,
        # choices=[86400, 125625, 13800],
        required=False,
        help="Choose last iteration of each dataset: 86400, 125625, or 13800",
    )
    # set model parameters
    parser.add_argument(
        "--model_type",
        type=str,
        choices=[
            "vae",
            "betavae",
            "controlvae",
            "factorvae",
            "betatcvae",
            "groupvae",
            "groupbetatcvae",
            "groupfactorvae",
            "groupcontrolvae",
            "commutativevae",
            "groupcommutativevae",
        ],
        required=True,
        help="choose vae type",
    )
    parser.add_argument(
        "--dense_dim",
        nargs="*",
        default=[256, 128],
        type=int,
        required=False,
        help="set CNN hidden FC layers",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=8,
        required=False,
        help="set prior dimension z",
    )
    # for model hyper-parameters
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        required=False,
        help="Set hyper-parameter alpha",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        required=False,
        help="Set hyper-parameter beta",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        required=False,
        help="Set hyper-parameter gamma",
    )
    parser.add_argument(
        "--lamb",
        type=float,
        default="1.0",
        required=False,
        help="Set hyper-parameter lambda",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default="1.0",
        required=False,
        help="Set hyper-parameter epsilon",
    )
    # Factor VAE hyper-parameters
    parser.add_argument(
        "--discri_lr_rate",
        type=float,
        default=4e-4,
        required=False,
        help="Set discriminator learning rate",
    )
    # Group--VAE
    parser.add_argument(
        "--th",
        type=float,
        default=0.1,
        required=False,
        help="Set threshold for pseudo labeling loss",
    )
    parser.add_argument(
        "--contrastive",
        type=str,
        default="barlow-twins",
        required=False,
        help="Choose Cotrastive Loss",
    )
    # Commutative-VAE
    parser.add_argument(
        "--hy_hes",
        type=float,
        default=40.0,
        required=False,
        help="Set hyper-parameter for commutative-VAE",
    )
    parser.add_argument(
        "--hy_rec",
        type=float,
        default=0.1,
        required=False,
        help="Set hyper-parameter for commutative-VAE",
    )
    parser.add_argument(
        "--hy_commute",
        type=float,
        default=20.0,
        required=False,
        help="Set hyper-parameter for commutative-VAE",
    )
    parser.add_argument(
        "--forward_eq_prob",
        type=float,
        default=0.2,
        required=False,
        help="Set hyper-parameter for commutative-VAE",
    )
    parser.add_argument(
        "--subgroup_sizes_ls",
        nargs="*",
        default=[100],
        type=int,
        required=False,
        help="Set hyper-parameter for commutative-VAE",
    )
    parser.add_argument(
        "--subspace_sizes_ls",
        nargs="*",
        default=[10],
        type=int,
        required=False,
        help="Set hyper-parmaeter for commutative-VAE",
    )
    # Control-VAE
    parser.add_argument(
        "--const_kld",
        default=16.0,
        type=float,
        required=False,
        help="set hyper-parameter for Control-VAE desired kl value",
    )
    parser.add_argument(
        "--max_beta",
        default=100.0,
        type=float,
        required=False,
        help="Set set hyper-parameter for Control-VAE max beta value",
    )
    parser.add_argument(
        "--k_i",
        default=0.001,
        type=float,
        required=False,
        help="Set set hyper-parameter for Control-VAE k_i for I controller",
    )
    parser.add_argument(
        "--k_p",
        default=0.01,
        type=float,
        required=False,
        help="Set set hyper-parameter for Control-VAE k_p for P controller",
    )
    parser.add_argument(
        "--no_exp",
        action="store_true",
    )
    parser.add_argument(
        "--num_sampling",
        type=int,
        default=1,
        required=False,
        help="Set hyper-parameter for samplings on TC-Beta-VAE",
    )
    parser.add_argument(
        "--lr_rate", default=1e-4, type=float, required=False, help="Set learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        required=False,
        help="Set weight decay",
    )
    # set training info
    parser.add_argument(
        "--split",
        type=float,
        default=0.2,
        required=False,
        help="set split ratio for train set and test set",
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="whether shuffling dataset or not"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=128,
        required=False,
        help="Set number of training mini-batch size",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size",
        type=int,
        default=128,
        required=False,
        help="Set number of training mini-batch size for multi GPU training",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=128,
        required=False,
        help="Set number of evaluation mini-batch size",
    )
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=60,
        required=False,
        help="Set number of epoch size",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        required=False,
        help="Set number of epoch size",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        required=False,
        help="Save model checkpoint iteration interval",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1000,
        required=False,
        help="Update tb_writer iteration interval",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        required=False,
        help="interval for early stopping",
    )
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "adam"],
        default="sgd",
        type=str,
        help="Choose optimizer",
        required=False,
    )
    parser.add_argument(
        "--scheduler",
        choices=["const", "linear"],
        default="const",
        type=str,
        help="Whether using scheduler during training or not",
        required=False,
    )
    parser.add_argument(
        "--num_disen_train",
        type=int,
        default=10,
        required=False,
        help="set number of disentanglement evaluation task",
    )
    parser.add_argument(
        "--num_disen_test",
        type=int,
        default=10,
        required=False,
        help="set number of disentanglement evaluation task",
    )
    parser.add_argument(
        "--batch_disen",
        type=int,
        default=100,
        required=False,
        help="set batch for Factor VAE disentanglement learning",
    )
    parser.add_argument(
        "--mask",
        type=float,
        required=False,
        default=0.0,
        help="for MIET-VAE but it is not used for group VAE",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument("--do_train", action="store_true", help="Do training")
    parser.add_argument("--do_eval", action="store_true", help="Do evaluation")
    parser.add_argument("--evaluate_during_training", action="store_true")
    parser.add_argument("--save_imgs", action="store_true", help="Do save imgs")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Whether write tensorboard or not",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        required=False,
        help="set seed",
    )
    parser.add_argument(
        "--sub_sec",
        type=int,
        default=10,
        required=True,
        help="set the number of elements in each section",
    )
    # for control vae
    parser.add_argument(
        "--t_total",
        type=int,
        default=1,
        required=False,
    )
    # qualitative analysis
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        required=False,
        help="Choose the interval for latent vector values",
    )
    parser.add_argument(
        "--quali_sampling",
        type=int,
        default=1,
        required=False,
        help="Set hyper-parameter for samplings on TC-Beta-VAE",
    )

    # for weight and bias writer
    parser.add_argument(
        "--project_name",
        type=str,
        required=True,
        help="set project name for wiehgt and bias writer",
    )

    # for multi FVM
    parser.add_argument(
        "--do_mfvm",
        action="store_true",
        help="Whether run tensorboard or not",
    )

    args = parser.parse_args()

    MODEL_CLASSES = {
        "betavae": (CNNBetaVAEConfig, CNNBetaVAE),
        "betatcvae": (CNNBetaTCVAEConfig, CNNBetaTCVAE),
        "commutativevae": (CNNLieConfig, CommutativeVAE),
        "controlvae": (ControlVAEConfig, CNNControlVAE),
        "groupvae": (GroupActionVAEConfig, CNNGroupVAE),
        "groupbetatcvae": (GroupActionBetaTCVAEConfig, CNNGroupBetaTCVAE),
        "groupcontrolvae": (GroupActionControlVAEConfig, CNNGroupControlVAE),
        "groupcommutativevae": (
            GroupActionCommutativeVAEConfig,
            CNNGroupCommutativeVAE,
        ),
    }

    # set VAE dense dim by dataset.
    args.dense_dim = DATA_HIDDEN_DIM[args.dataset]

    # with torch.cuda.device(args.device_idx):
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    data_loader = DATALOADER[args.dataset](
        path=args.data_dir,
        shuffle_dataset=True,
        random_seed=args.seed,
        split_ratio=args.split,
    )  # datasets[args.dataset](args)

    dataset_size = len(data_loader)

    train_dataloader = DataLoader(
        data_loader, batch_size=args.train_batch_size, drop_last=True, pin_memory=True
    )
    t_total = (
        len(train_dataloader) * args.num_epoch
        if args.max_steps == 0
        else args.max_steps
    )
    length = len(train_dataloader)
    args.num_epoch = (
        args.num_epoch if args.max_steps == 0 else args.max_steps // length + 1
    )
    args.t_total = t_total

    config, model = MODEL_CLASSES[args.model_type]
    config = (
        config(args=args, dataset_size=dataset_size)
        if "betatcvae" in args.model_type
        else config(args=args)
    )
    model = model(config=config)
    model.init_weights()

    train_dataloader = None  # remove train_dataloader

    # Only for evaluation
    if args.do_train != True and args.do_eval:
        save_file = make_run_files(args)
        args.model_dir = args.output_dir
        args.steps = DATA_STEPS[args.dataset]
        sub_model, path = load_model(args, save_file=save_file)
        if os.path.exists(path):
            model.load_state_dict(sub_model)

    model.to(device)
    loss_fn = (
        nn.BCELoss(reduction="sum")
        if args.dataset == "dsprites" or args.dataset == "cdsprites"
        else nn.MSELoss(reduction="sum")
    )

    results = None

    if args.do_train and args.do_eval == False:
        train(
            train_dataset=data_loader,
            num_epochs=args.num_epoch,
            model=model,
            loss_fn=loss_fn,
            args=args,
        )

    elif args.do_eval and args.do_train == False:
        results = eval(
            train_dataset=data_loader, model=model, loss_fn=loss_fn, args=args
        )

    elif args.do_train and args.do_eval:
        train(
            train_dataset=data_loader,
            num_epochs=args.num_epoch,
            model=model,
            loss_fn=loss_fn,
            args=args,
        )

        results = eval(
            train_dataset=data_loader, model=model, loss_fn=loss_fn, args=args
        )

    # Qualitative Analysis
    save_file = make_run_files(args)
    path = os.path.join(args.output_dir, args.model_type, save_file)
    reconstruction_imgs = qualitative(dataset=data_loader, model=model, args=args)
    imgs_dir = os.path.join(path, "images.png")
    if (
        args.dataset == "dsprites"
        or args.dataset == "shapes3d"
        or args.dataset == "smallnorb"
    ):
        plotting_3d(data_loader, model, loss_fn, save_file, args)
    save_image(reconstruction_imgs, imgs_dir, nrow=args.interval + 1, pad_value=1.0)
    # extract basis, attention distribution and symmetries

    comparing_baseline(data_loader, model, loss_fn, save_file, args)
    analysis_of_basis(data_loader, model, loss_fn, save_file, args)

    if "group" in args.model_type:
        extract_group_actions(data_loader, model, loss_fn, save_file, args)

    # Quantitative Analysis
    if args.dataset == "celeba":
        return

    else:
        # Disentanglement Metric (FVM, SAP, MIG, DCI)
        disent_results = estimate_all_distenglement(
            data_loader,
            model,
            disent_batch_size=args.batch_disen,
            disent_num_train=args.num_disen_train,
            disent_num_test=args.num_disen_test,
            loss_fn=loss_fn,
            continuous_factors=False,
            args=args,
        )

        results["beta_vae"] = disent_results["beta_vae"]
        results["factor_disent"] = disent_results["factor"]["disentanglement_accuracy"]
        results["mig"] = disent_results["mig"]
        results["sap"] = disent_results["sap"]
        results["dci_disent"] = disent_results["dci"]["disent"]
        results["dci_comple"] = disent_results["dci"]["comple"]
        results["mfvm_2"] = disent_results["mfvm_2"]
        if "mfvm_3" in disent_results.keys():
            results["mfvm_3"] = disent_results["mfvm_3"]
        if "mfvm_4" in disent_results.keys():
            results["mfvm_4"] = disent_results["mfvm_4"]
        if "mfvm_5" in disent_results.keys():
            results["mfvm_5"] = disent_results["mfvm_5"]

        # save_files = make_run_files(args)
        output_dir = os.path.join(args.output_dir, args.model_type)
        if args.do_train and args.do_eval and not args.do_mfvm:
            args.results_file = os.path.join(output_dir, "result.csv")
        elif args.do_eval and not args.do_mfvm:
            args.results_file = os.path.join(output_dir, "eval_only_result.csv")
        elif args.do_train and args.do_eval and args.do_mfvm:
            args.results_file = os.path.join(output_dir, "result_full_wmfvm.csv")
        elif args.do_mfvm and not args.do_train and args.do_eval:
            args.results_file = os.path.join(output_dir, "eval_full_mfvm_results.csv")
        write_info(args, results)
    return


if __name__ == "__main__":
    main()
