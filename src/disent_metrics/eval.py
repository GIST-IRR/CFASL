import torch
from src.seed import set_seed
from src.disent_metrics.fvm import FactorVAEMetric
from src.disent_metrics.sap import SAP
from src.disent_metrics.dci import metric_dci
from src.disent_metrics.mig import MIGMetric
from src.disent_metrics.betavae import compute_beta_vae
from src.disent_metrics.utils import latents_and_factors
from src.disent_metrics.mfvm import mFVM
def estimate_all_distenglement(data_loader,
                               model,
                               disent_batch_size,
                               disent_num_train,
                               disent_num_test,
                               loss_fn,
                               continuous_factors,
                               args):

    with torch.no_grad():
        model.eval()
        results = {}

        if args.do_mfvm:
            fixed_num = 0
            if args.dataset == 'car':
                fixed_num = 1
            elif args.dataset == 'smallnorb':
                fixed_num = 2
            elif args.dataset == 'dsprites':
                fixed_num = 3
            elif args.dataset == 'shapes3d':
                fixed_num = 4
            #set_seed(args)
            for i in range(fixed_num):
                disent_result = mFVM(data_loader,
                                     model=model,
                                     batch_size=100,
                                     num_train=800,
                                     loss_fn=loss_fn,
                                     fixed_idx_num=i+2,
                                     args=args)
                key = 'mfvm_' + str(i+2)
                results[key] = disent_result['disentanglement_accuracy']

        # else:
        set_seed(args)
        disent_result = compute_beta_vae(data_loader,
                                         model=model,
                                         batch_size=64,
                                         num_train=500,
                                         num_eval=50,
                                         loss_fn=loss_fn,
                                         args=args)

        results['beta_vae'] = disent_result

        set_seed(args)
        disent_result = FactorVAEMetric(data_loader,
                                        model=model,
                                        batch_size=100,
                                        num_train=800,
                                        loss_fn=loss_fn,
                                        args=args)
        results['factor'] = disent_result


        set_seed(args)
        train_latents, train_factors = latents_and_factors(dataset=data_loader, model=model, batch_size=64, interation=100, loss_fn=loss_fn, args=args)
        test_latents, test_factors = latents_and_factors(dataset=data_loader, model=model, batch_size=64, interation=50, loss_fn=loss_fn, args=args)

        set_seed(args)
        disent_result = SAP(train_latents, train_factors, test_latents, test_factors, args, continuous_factors=continuous_factors)
        results['sap'] = disent_result

        set_seed(args)
        disent_result = metric_dci(train_latents, train_factors, test_latents, test_factors, args, continuous_factors=continuous_factors)
        results['dci'] = {}
        results['dci']['train_err'] = disent_result[0]
        results['dci']['test_err'] = disent_result[1]
        results['dci']['disent'] = disent_result[2]
        results['dci']['comple'] = disent_result[3]

        set_seed(args)
        disent_result = MIGMetric(train_latents, train_factors)
        results['mig'] = disent_result

    return results

