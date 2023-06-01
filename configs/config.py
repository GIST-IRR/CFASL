from configs.utils import VAEConfig


class GroupActionVAEConfig(VAEConfig):
    def __init__(self, args):
        super(GroupActionVAEConfig, self).__init__(args)
        self.contrastive = args.contrastive
        self.lamb = args.lamb
        self.alpha = args.alpha
        self.beta = args.beta
        self.th = args.th
        self.sub_sec = args.sub_sec


class GroupActionBetaTCVAEConfig(GroupActionVAEConfig):
    def __init__(self, args, dataset_size=0):
        super(GroupActionBetaTCVAEConfig, self).__init__(args)
        self.dataset_size = dataset_size


class GroupActionControlVAEConfig(GroupActionVAEConfig):
    def __init__(self, args):
        super(GroupActionControlVAEConfig, self).__init__(args)
        self.t_total = args.t_total
        self.const_kld = args.const_kld
        self.min_beta = args.beta
        self.max_beta = args.max_beta
        self.k_i = args.k_i
        self.k_p = args.k_p


class GroupActionCommutativeVAEConfig(GroupActionVAEConfig):
    def __init__(self, args):
        super(GroupActionCommutativeVAEConfig, self).__init__(args)
        self.subspace_sizes_ls = args.subspace_sizes_ls  # list of int
        self.subgroup_sizes_ls = args.subgroup_sizes_ls  # list of int
        self.no_exp = args.no_exp
        self.hy_hes = args.hy_hes
        self.hy_rec = args.hy_rec
        self.hy_commute = args.hy_commute
        self.forward_eq_prob = args.forward_eq_prob
