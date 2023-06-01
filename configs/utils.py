class VAEConfig:
    def __init__(self, args, hidden_states=True):
        self.mask = args.mask
        self.dataset = args.dataset
        self.dense_dim = args.dense_dim
        self.latent_dim = args.latent_dim
        self.hidden_states = hidden_states
        self.num_sampling = args.num_sampling


class CNNVanillaVAEConfig(VAEConfig):
    def __init__(self, args, in_channel=1):
        super(CNNVanillaVAEConfig, self).__init__(args)
        self.in_channel = in_channel


class CNNBetaVAEConfig(VAEConfig):
    def __init__(self, args, in_channel=1):
        super(CNNBetaVAEConfig, self).__init__(args)
        self.in_channel = in_channel
        self.alpha = args.alpha
        self.beta = args.beta
        self.lamb = args.lamb


class CNNBetaTCVAEConfig(VAEConfig):
    def __init__(self, args, in_channel=1, dataset_size=0):
        super(CNNBetaTCVAEConfig, self).__init__(args)
        self.in_channel = in_channel
        self.dataset_size = dataset_size


class ControlVAEConfig(VAEConfig):
    def __init__(self, args):
        super(ControlVAEConfig, self).__init__(args)
        self.t_total = args.t_total
        self.const_kld = args.const_kld
        self.min_beta = args.beta
        self.max_beta = args.max_beta
        self.k_i = args.k_i
        self.k_p = args.k_p


class CNNLieConfig(VAEConfig):
    def __init__(self, args, in_channel=1):
        super(CNNLieConfig, self).__init__(args)
        self.subspace_sizes_ls = args.subspace_sizes_ls  # list of int
        self.subgroup_sizes_ls = args.subgroup_sizes_ls  # list of int
        self.no_exp = args.no_exp
        self.hy_hes = args.hy_hes
        self.hy_rec = args.hy_rec
        self.hy_commute = args.hy_commute
        self.forward_eq_prob = args.forward_eq_prob
