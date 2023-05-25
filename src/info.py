import os
import csv
import argparse

def write_info(args, results):
    info = None
    if args.model_type == "vae":
        info = VanillaInfo(args, **results)
    elif args.model_type == "betavae":
        info = BetaInfo(args, **results)
    elif args.model_type == "betatcvae":
        info = BetaTCInfo(args, **results)
    elif args.model_type == "controlvae":
        info = ControlInfo(args, **results)
    elif args.model_type == "commutativevae":
        info = CommutativeInfo(args, **results)
    elif args.model_type =="groupvae":
        info = GroupInfo(args, **results)
    elif args.model_type =="groupbetatcvae":
        info = GroupBetaTCInfo(args, **results)
    elif args.model_type =="groupcontrolvae":
        info = GroupControlInfo(args, **results)
    elif args.model_type == "groupcommutativevae":
        info = GroupCommutativeInfo(args, **results)
    info.write_results()
    return

class VanillaInfo():

    def __init__(self, args, **kwargs):
        self.file_dir = args.results_file
        self.opt = args.optimizer
        self.epoch = args.num_epoch
        self.lr = args.lr_rate
        self.seed = args.seed
        self.wd = args.weight_decay
        self.batch = args.train_batch_size
        self.latent = args.latent_dim
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.lamb = args.lamb
        self.mask = args.mask
        self.epsilon = None #args.epsilon
        self.elbo = kwargs['elbo']
        #self.obj = kwargs['obj']

        self.reconst = kwargs['reconst']
        self.kld = kwargs['kld']
        self.beta_vae = kwargs['beta_vae']
        self.factor_disent = kwargs['factor_disent']
        self.mig = kwargs['mig']
        self.sap = kwargs['sap']
        self.dci_disent = kwargs['dci_disent']
        self.dci_completness = kwargs['dci_comple']

        if 'group' in args.model_type and not args.do_mfvm:
            self.attn_beta = kwargs['attn_beta']
        if args.do_mfvm:
            self.mfvm_2 = kwargs['mfvm_2']
            if 'mfvm_3' in kwargs.keys():
                self.mfvm_3 = kwargs['mfvm_3']
            if 'mfvm_4' in kwargs.keys():
                self.mfvm_4 = kwargs['mfvm_4']
            if 'mfvm_5' in kwargs.keys():
                self.mfvm_5 = kwargs['mfvm_5']
        # if args.do_mfvm and 'group' in args.model_type:
        #     self.bet = kwargs['bet']
        #     self.int = kwargs['int']
        #elif args.do_mfvm == False:

        #self.disen_acc = kwargs['disen_acc']
        #self.act_dim = kwargs['act_dims']

    def write_results(self):

        file_exists = os.path.isfile(self.file_dir)
        fieldnames = [str(key) for key in self.__dict__]

        with open(self.file_dir, 'a+', newline='') as f:
            writer = csv.DictWriter(f, fieldnames= fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(self.__dict__)
        return

class Eval_Info(VanillaInfo):
    def __init__(self, args, **kwargs):
        super(Eval_Info, self).__init__(args, **kwargs)
        self.elbo = kwargs['elbo']
        self.obj = kwargs['obj']
        self.reconst = kwargs['reconst']
        self.kld = kwargs['kld']
        self.dci_train_err = kwargs['dci_train_err']
        self.dci_eval_err = kwargs['dci_eval_err']

class BetaInfo(VanillaInfo):

    def __init__(self, args, **kwargs):
        super(BetaInfo, self).__init__(args, **kwargs)
        self.beta = args.beta

class BetaTCInfo(VanillaInfo):

    def __init__(self, args, **kwargs):
        super(BetaTCInfo, self).__init__(args, **kwargs)
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.tc = kwargs['tc']
        self.mi = kwargs['mi']

class ControlInfo(VanillaInfo):
    def __init__(self, args, **kwargs):
        super(ControlInfo, self).__init__(args, **kwargs)
        self.const_kld = args.const_kld
        self.max_beta = args.max_beta
        self.k_p = args.k_p
        self.k_i = args.k_i

class CommutativeInfo(VanillaInfo):

    def __init__(self, args, **kwargs):
        super(CommutativeInfo, self).__init__(args, **kwargs)
        self.hy_hes = args.hy_hes
        self.hy_rec = args.hy_rec
        self.hy_commute = args.hy_commute
        self.forward_eq_prob = args.forward_eq_prob
        self.group = kwargs['group']

class GroupInfo(VanillaInfo):
    def __init__(self, args, **kwargs):
        super(GroupInfo, self).__init__(args, **kwargs)
        self.alpha = args.alpha
        self.epsilon = args.epsilon
        self.th = args.th
        self.sub_sec = args.sub_sec


class GroupBetaTCInfo(GroupInfo):
    def __init__(self, args, **kwargs):
        super(GroupBetaTCInfo, self).__init__(args, **kwargs)
        self.tc = kwargs['tc']
        self.mi = kwargs['mi']

class GroupControlInfo(GroupInfo):
    def __init__(self, args, **kwargs):
        super(GroupControlInfo, self).__init__(args, **kwargs)
        self.const_kld = args.const_kld
        self.max_beta = args.max_beta
        self.k_p = args.k_p
        self.k_i = args.k_i

class GroupCommutativeInfo(GroupInfo):
    def __init__(self, args, **kwargs):
        super(GroupCommutativeInfo, self).__init__(args, **kwargs)
        self.hy_hes = args.hy_hes
        self.hy_rec = args.hy_rec
        self.hy_commute = args.hy_commute
        self.forward_eq_prob = args.forward_eq_prob
        self.subgroup_sizes_ls = args.subgroup_sizes_ls
        self.subspace_sizes_ls = args.subspace_sizes_ls
