def make_run_files(args):
    if args.model_type == "mpl" or args.model_type == "vae":
        file = "opt:{}_epoch:{}_lr:{}_seed:{}_wd:{}_batch:{}_dim:{}_vae".format(args.optimizer,
                                                                                args.num_epoch,
                                                                                args.lr_rate,
                                                                                args.seed,
                                                                                args.weight_decay,
                                                                                args.train_batch_size,
                                                                                args.latent_dim)
        return file

    elif args.model_type == "betavae":
        file = "opt:{}_epoch:{}_lr:{}_seed:{}_wd:{}_batch:{}_beta:{}_dim:{}_beta".format(args.optimizer,
                                                                                         args.num_epoch,
                                                                                         args.lr_rate,
                                                                                         args.seed,
                                                                                         args.weight_decay,
                                                                                         args.train_batch_size,
                                                                                         args.beta,
                                                                                         args.latent_dim)
        return file

    elif args.model_type == "betatcvae":
        file = "opt:{}_epoch:{}_lr:{}_seed:{}_wd:{}_batch:{}_alpha:{}_beta:{}_labmda:{}_dim:{}_betatc".format(args.optimizer,
                                                                                                    args.num_epoch,
                                                                                                    args.lr_rate,
                                                                                                    args.seed,
                                                                                                    args.weight_decay,
                                                                                                    args.train_batch_size,
                                                                                                    args.alpha,
                                                                                                    args.beta,
                                                                                                    args.lamb,
                                                                                                    args.latent_dim)
        return file

    elif args.model_type == "controlvae":
        file = "opt:{}_epoch:{}_lr:{}_seed:{}_wd:{}_batch:{}_kld:{}_minbeta:{}_maxbeta:{}_ki:{}_kp:{}_dim:{}_mask:{}_control".format(args.optimizer,
                                                                                                                            args.num_epoch,
                                                                                                                            args.lr_rate,
                                                                                                                            args.seed,
                                                                                                                            args.weight_decay,
                                                                                                                            args.train_batch_size,
                                                                                                                            args.const_kld,
                                                                                                                            args.beta,
                                                                                                                            args.max_beta,
                                                                                                                            args.k_i,
                                                                                                                            args.k_p,
                                                                                                                            args.latent_dim,
                                                                                                                              args.mask)
        return file

    elif args.model_type =="commutativevae":
        file = "opt:{}_epoch:{}_lr:{}_seed:{}_wd:{}_batch:{}_dim:{}_hes:{}_commu:{}_rec:{}_mask:{}_commtative".format(args.optimizer,
                                                                                                args.num_epoch,
                                                                                                args.lr_rate,
                                                                                                args.seed,
                                                                                                args.weight_decay,
                                                                                                args.train_batch_size,
                                                                                                args.latent_dim,
                                                                                                args.hy_hes,
                                                                                                args.hy_commute,
                                                                                                args.hy_rec,
                                                                                                                              args.mask)
        return file

    elif args.model_type == "groupvae":
        file = "opt:{}_epoch:{}_lr:{}_seed:{}_wd:{}_batch:{}_dim:{}_alpha:{}_beta:{}_gamma:{}_lamb:{}_eps:{}_th:{}_subsec:{}_vae".format(args.optimizer,
                                                                                                                                args.num_epoch,
                                                                                                                                args.lr_rate,
                                                                                                                                args.seed,
                                                                                                                                args.weight_decay,
                                                                                                                                args.train_batch_size,
                                                                                                                                args.latent_dim,
                                                                                                                                args.alpha,
                                                                                                                                args.beta,
                                                                                                                                args.gamma,
                                                                                                                                args.lamb,
                                                                                                                                args.epsilon,
                                                                                                                                args.th,
                                                                                                                                args.sub_sec)
        return file

    elif args.model_type =="groupbetatcvae":
        file = "opt:{}_epoch:{}_lr:{}_seed:{}_wd:{}_batch:{}_dim:{}_alpha:{}_beta:{}_gamma:{}_lamb:{}_eps:{}_th:{}_subsec:{}_betatc".format(args.optimizer,
                                                                                                                                    args.num_epoch,
                                                                                                                                    args.lr_rate,
                                                                                                                                    args.seed,
                                                                                                                                    args.weight_decay,
                                                                                                                                    args.train_batch_size,
                                                                                                                                    args.latent_dim,
                                                                                                                                    args.alpha,
                                                                                                                                    args.beta,
                                                                                                                                    args.gamma,
                                                                                                                                    args.lamb,
                                                                                                                                    args.epsilon,
                                                                                                                                    args.th,
                                                                                                                                    args.sub_sec)
        return file

    elif args.model_type == "groupcontrolvae":
        file = "opt:{}_epoch:{}_lr:{}_seed:{}_wd:{}_batch:{}_dim:{}_alpha{}_gamma:{}_lamb:{}_eps:{}_th:{}_subsec:{}_kld:{}_minbeta:{}_maxbeta:{}_ki:{}_kp:{}_control".format(args.optimizer,
                                                                                                                                                    args.num_epoch,
                                                                                                                                                    args.lr_rate,
                                                                                                                                                    args.seed,
                                                                                                                                                    args.weight_decay,
                                                                                                                                                    args.train_batch_size,
                                                                                                                                                    args.latent_dim,
                                                                                                                                                    args.alpha,
                                                                                                                                                    args.gamma,
                                                                                                                                                    args.lamb,
                                                                                                                                                    args.epsilon,
                                                                                                                                                    args.th,
                                                                                                                                                    args.sub_sec,
                                                                                                                                                     args.const_kld,
                                                                                                                                                     args.beta,
                                                                                                                                                     args.max_beta,
                                                                                                                                                     args.k_i,
                                                                                                                                                     args.k_p)
        return file

    elif args.model_type =="groupcommutativevae":
        file = "opt:{}_epoch:{}_lr:{}_seed:{}_wd:{}_batch:{}_dim:{}_alpha{}_gamma:{}_lamb:{}_eps:{}_th:{}_subsec:{}_hes:{}_commu:{}_rec:{}_commtative".format(args.optimizer,
                                                                                                                                                        args.num_epoch,
                                                                                                                                                        args.lr_rate,
                                                                                                                                                        args.seed,
                                                                                                                                                        args.weight_decay,
                                                                                                                                                        args.train_batch_size,
                                                                                                                                                        args.latent_dim,
                                                                                                                                                        args.alpha,
                                                                                                                                                        args.gamma,
                                                                                                                                                        args.lamb,
                                                                                                                                                        args.epsilon,
                                                                                                                                                        args.th,
                                                                                                                                                        args.sub_sec,
                                                                                                                                                        args.hy_hes,
                                                                                                                                                        args.hy_commute,
                                                                                                                                                        args.hy_rec)
        return file