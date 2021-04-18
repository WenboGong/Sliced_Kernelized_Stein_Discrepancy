import sys
import os
import argparse
import warnings


import torch
import numpy as np

from tqdm import tqdm

cwd=os.getcwd()
cwd_parent=os.path.abspath('..')
sys.path.append(cwd)
sys.path.append(cwd_parent+'/src')
sys.path.append(cwd_parent)

from src.Util import *
from src.Divergence import *
from src.Kernel import *
from src.distribution import *
from src.GOF_Test import *
from src.active_slice import *
from src.Dataloader import *
import time
import pickle

parser=argparse.ArgumentParser(description='RBM GOF Test')
parser.add_argument('--n_trials',type=int,default=100)
parser.add_argument('--n_samples',type=int,default=1000)
parser.add_argument('--burn_in_samples',type=int,default=2000) # pseudo samples used for training during burn-in.
# Note: we should not merge them with test samples as they are from burn-in stage.
parser.add_argument('--pseudo_setting',type=str,default='Setting1') # Setting2: train r,g during burn-in (SKSD paper setting)
# Setting1: Collecting pseudo samples first during burn-in, then train r,g using them (active slice paper setting)

parser.add_argument('--burnin',type=int,default=2000) # Burn-in period
parser.add_argument('--perturb', metavar='pertur', type=float, nargs='*',
                        default=[0,0.005,0.007,0.009,0.01,0.011,0.012,0.013,0.014,0.016,0.018]) # Perturbation level list
parser.add_argument('--num_select', type=int,
                        default=50) # pruning level: how many r/g we keep.

parser.add_argument('--batch_size',type=int,default=100) # batch size for GO
parser.add_argument('--grad_opt_ep',type=int,default=100) # GO epoch
parser.add_argument('--method_score_q',type=str,default='kernel_smooth') # gradient estimator or exact
parser.add_argument('--method_r_opt',type=str,default='Not') # Fixed or Not. Fixed: r is not updated. Not: r is also updated
parser.add_argument('--optimal_init', action='store_true') # Use this to activate the active slice method for initialization
parser.add_argument('--optimal_grad', action='store_true') # Use this to activate GO for slice direction.
# Note at least use one of them. E.g. set --optimal_init to enable just active slice, set --optimal_grad to use just GO, and set both to use active slice for initialization and then GO refinement.
parser.add_argument('--method',type=str,default='SKSD') # KSD or MMD
parser.add_argument('--disable_gpu',action='store_true') # set this to disable gpu


args=parser.parse_args()
flag_optimal_init = args.optimal_init
flag_optimal_grad = args.optimal_grad
pseudo_setting = args.pseudo_setting

path='...' # Change to your own WDR

dtype = torch.FloatTensor
if not args.disable_gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    dtype = torch.cuda.FloatTensor

torch.set_default_tensor_type(dtype)


# Store results
result_dict={}
result_dict['result'] = {}

# Hyperparameter settings
n_trials=args.n_trials
sample_size_full=args.n_samples
significance=0.05
n_boots=1000
burnin=args.burnin
perturb_level_list=args.perturb
dx=50 # RBM dimensions
dh=40 # Hidden variable dimension
grad_opt_ep=args.grad_opt_ep
num_select=args.num_select
burn_in_tr_sample=args.burn_in_samples
batch_size=args.batch_size
method=args.method # KSD or MMD
method_score_q=args.method_score_q
method_r_opt=args.method_r_opt

# Store hyperparameter settings
result_dict=para_store(result_dict,n_trials=args.n_trials,
n_samples=args.n_samples,
significance=significance,
n_boots=n_boots,
burnin=args.burnin,
perturb_level_list=args.perturb,
dx_list=dx,
dh_list=dh,
batch_size=args.batch_size,
bandwidth_Power_list=0.5,
flag_optimal_init=flag_optimal_init,
flag_optimal_burn=True,
flag_optimal_grad=flag_optimal_grad,burn_in_tr_sample=burn_in_tr_sample,method=method,
grad_opt_ep=args.grad_opt_ep,
num_select=args.num_select
                       )

######## Debugs
# pseudo_setting='Setting1'
# num_select = 3
# perturb_level_list=[0.01]
# flag_optimal_init=True
# flag_opt_r='Not'
# flag_optimal_grad=False
# method_score_q = 'kernel_smooth'
# grad_opt_ep=10
# ##################

for p_level in perturb_level_list:
    # For storing results
    SKSD_results = np.zeros((n_trials))
    KSD_results = np.zeros((n_trials))
    MMD_results = np.zeros((n_trials))

    KSD_value = np.zeros((n_trials))
    SKSD_value = np.zeros((n_trials))
    MMD_value = np.zeros((n_trials))

    SKSD_correct = np.zeros(n_trials)
    KSD_correct = np.zeros(n_trials)
    MMD_correct = np.zeros(n_trials)

    result_dict['result']['p_level:%s' % (p_level)] = {}

    # Fix random seed
    torch.manual_seed(2)
    np.random.seed(2)
    print('Method:%s Perturb Level:%s' % (method,p_level))
    if flag_optimal_grad:
        print('grad_Opt_ep:%s' % grad_opt_ep)


    # Define RBM model
    b = np.random.randn(dx)
    c = np.random.randn(dh)
    B = np.random.choice([-1, 1], size=[dx, dh])

    Bcorrupted = B + p_level * np.random.randn(dx, dh)

    b_pt = torch.from_numpy(b).type(dtype)
    c_pt = torch.from_numpy(c).type(dtype)
    B_pt = torch.from_numpy(B).type(dtype)
    Bcorrupted_pt = torch.from_numpy(Bcorrupted).type(dtype)

    # Define distribution
    p_x = GaussBernRBM(B_pt, b_pt, c_pt)
    p_x_ds = DSGaussBernRBM_GPU(B_pt, b_pt, c_pt)
    H_1_q = GaussBernRBM(Bcorrupted_pt, b_pt, c_pt)
    H_1_q_ds = DSGaussBernRBM_GPU(Bcorrupted_pt, b_pt, c_pt)

    # For computing time
    time_optimal_init = 0
    time_grad = 0
    time_KSD_avg = 0
    time_SKSD_avg = 0
    time_SKSD_init_avg = 0
    time_SKSD_grad_avg = 0

    # Start the trials
    for idx_trial in tqdm(range(n_trials)):
        # Randomly picking r,g with eye initialization
        idx_se = torch.randperm(50)[0:num_select]
        r = torch.eye(dx, dx)[idx_se, :]
        g = torch.eye(dx, dx)[idx_se, :]

        kernel_hyper_SKSD = {
            'bandwidth': None
        }
        q_samples, q_samples_burn = H_1_q_ds.sample(sample_size_full, burnin=burnin, burn_samples=True,
                                                    burn_in_tr_sample=burn_in_tr_sample)
        q_samples = q_samples.requires_grad_()

        p_samples = p_x_ds.sample(sample_size_full, burnin=burnin)

        if method == 'SKSD':

            if pseudo_setting == 'Setting2':

                test_sample = q_samples[0:sample_size_full-200+1,:] # test size = sample_size-batchsize, because we train during burn-in, so the last burn-in step has batch size 200,
                # we treat it as the true samples.

                time_start_grad = time.time()
                flag_opt_r = True if method_r_opt=='Not' else False
                tr_rg,H = H_1_q_ds.sample(200,return_latent=True,burnin=1) # initial pseudo samples
                tr_rg.requires_grad_()
                # train r,g during the burnin-period, so each burn in samples 200 pseudo samples and update r,g. Total 2000 burn in steps.
                g, r = compute_optimal_g_MCMC(burnin, dx, p_x.log_prob, SE_kernel,g=g,batch_size=200,
                                                           r=r,flag_U=True, median_power=0.5,
                                                           kernel_hyper=kernel_hyper_SKSD,
                                                           dd_kernel=dd_SE_kernel,
                                                           d_kernel=d_SE_kernel, fix_sample=(tr_rg, H),
                                                           flag_optimal_r=flag_opt_r,
                                                           q_x_ds=H_1_q_ds,update_interval=1
                                                           )
                time_grad = time.time() - time_start_grad

            elif pseudo_setting == 'Setting1':

                # Split into train and test

                tr_sample = q_samples_burn #these are the pseudo samples
                test_sample = q_samples #these are the true test samples
                ## If enable optimal init
                if flag_optimal_init:
                    if method_score_q == 'gradient_estimator':
                        time_start = time.time()
                        with torch.no_grad():
                            median_dist = median_heruistic(tr_sample, tr_sample.clone())
                            bandwidth = 2 * torch.pow(0.5 * median_dist, 0.5)
                        tr_sample.requires_grad_()
                        score_q_esti = gradient_estimate_im(tr_sample, bandwidth, lam=0.01)
                        if method_r_opt != 'Fixed':
                            r = r_optimal(tr_sample, p_x.log_prob, None, threshold=None, num_selection=num_select,
                                          score_q=score_q_esti)

                        g = Poincare_g_optimal_SVD(tr_sample, p_x.log_prob, None, r=r, score_q=score_q_esti)
                        time_optimal_init = time.time() - time_start
                    elif method_score_q == 'kernel_smooth':
                        time_start = time.time()
                        if method_r_opt != 'Fixed':
                            r = r_kernel(tr_sample, tr_sample, p_x.log_prob, threshold=None, num_selection=num_select,
                                         fix_sample=True)
                        g = Poincare_g_kernel_SVD(tr_sample, tr_sample, p_x.log_prob, r=r, fix_sample=True, einsum=True,
                                                  lobpcg=False)
                        time_optimal_init = time.time() - time_start
                    elif method_score_q == 'exact':
                        time_start = time.time()

                        tr_sample.requires_grad_()
                        if method_r_opt != 'Fixed':
                            r = r_optimal(tr_sample, p_x.log_prob, H_1_q.log_prob, threshold=None,
                                          num_selection=num_select)
                        g = Poincare_g_optimal_SVD(tr_sample, p_x.log_prob, H_1_q.log_prob, r=r)
                        time_optimal_init = time.time() - time_start

                # Enable GO optimization
                if flag_optimal_grad:
                    ######################## gradient based optimization ######################

                    r_init = r.clone().detach().requires_grad_()
                    Adam_r = torch.optim.Adam([r_init], lr=0.001, betas=(0.9, 0.99))

                    tr_sample.requires_grad_()
                    r_n = r_init

                    g_init = g.clone().detach().requires_grad_()

                    Adam_g = torch.optim.Adam([g_init], lr=0.001, betas=(0.9, 0.99))

                    g_n = g_init / (torch.norm(g_init, 2, dim=-1, keepdim=True) + 1e-10)
                    # Define data loader
                    tr_dataset = Sample_Dataset(tr_sample.clone().detach())
                    loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)

                    time_start_grad = time.time()
                    for i in range(grad_opt_ep):
                        for j, data in enumerate(loader):
                            Adam_r.zero_grad()
                            Adam_g.zero_grad()
                            kernel_hyper_maxSVGD = {
                                'bandwidth': None
                            }
                            data.requires_grad_()
                            start_time = time.time()
                            score1 = torch.autograd.grad(p_x.log_prob(data).sum(), data)[0]
                            diver, _ = compute_max_DSSD_eff(data, data.clone().detach(), None, SE_kernel,
                                                            d_kernel=d_SE_kernel,
                                                            dd_kernel=dd_SE_kernel,
                                                            r=r_n, g=g_n, kernel_hyper=kernel_hyper_maxSVGD,
                                                            score_samples1=score1,
                                                            score_samples2=score1.clone().detach()
                                                            , flag_median=True, flag_U=False, median_power=0.5,
                                                            bandwidth_scale=1
                                                            )

                            (-diver).backward()
                            if method_r_opt != 'Fixed':
                                Adam_r.step()
                            Adam_g.step()
                            r_n = r_init / (torch.norm(r_init, 2, dim=-1, keepdim=True) + 1e-10)
                            g_n = g_init / (torch.norm(g_init, 2, dim=-1, keepdim=True) + 1e-10)

                    time_grad = time.time() - time_start_grad
            else:
                raise NotImplementedError('Choose either Setting1 or Setting2')
            # re-normalize
            g = g / (torch.norm(g, 2, dim=-1, keepdim=True) + 1e-10)
            r = r / (torch.norm(r, 2, dim=-1, keepdim=True) + 1e-10)
            g,r=g.detach(),r.detach()

            ############ SKSD Test ########################
            time_start = time.time()
            SKSD, SKSD_bootstrap = bootstrap_SKSD(n_boots, p_x, test_sample, r, g,
                                                  SE_kernel, kernel_hyper=kernel_hyper_SKSD,
                                                  median_power=0.5, d_kernel=d_SE_kernel,
                                                  dd_kernel=dd_SE_kernel, split=1
                                                  )

            prop_SKSD = torch.mean((SKSD_bootstrap >= SKSD).float().type(dtype))
            time_SKSD = time.time() - time_start + time_optimal_init + time_grad
            SKSD_value[idx_trial] = SKSD.cpu().data.numpy()
            if prop_SKSD < significance:
                # rejection
                SKSD_correct[idx_trial] = 1.
            else:
                SKSD_correct[idx_trial] = 0.

            # time consumption
            time_SKSD_avg += time_SKSD
            time_SKSD_grad_avg += time_grad
            time_SKSD_init_avg += time_optimal_init
            if (idx_trial + 1) % 1 == 0:
                print(
                    'Dx:%s Per:%s trial:%s SKSD:%s SKSD_v:%s prop_SKSD:%s time_SKSD:%s ' % (
                        dx, p_level, idx_trial, SKSD_correct[0:idx_trial + 1].sum() / (idx_trial + 1.),
                        SKSD.cpu().data.numpy(), prop_SKSD.cpu().data.numpy(),
                        time_SKSD_avg / (idx_trial + 1.)))
            result_dict['result']['p_level:%s' % (p_level)]['time_SKSD_avg'] = time_SKSD_avg / (idx_trial + 1.)
            result_dict['result']['p_level:%s' % (p_level)]['time_SKSD_init_avg'] = time_SKSD_init_avg / (idx_trial + 1.)
            result_dict['result']['p_level:%s' % (p_level)]['time_SKSD_grad_avg'] = time_SKSD_grad_avg / (idx_trial + 1.)
        elif method == 'KSD':
            ############ KSD Test ##########################
            start_time = time.time()
            if q_samples.shape[0] > 500:
                idx_crop = 500
            else:
                idx_crop = q_samples.shape[0]
            median_dist = median_heruistic(q_samples[0:idx_crop, :], q_samples[0:idx_crop, :].clone())
            bandwidth = 2 * torch.pow(0.5 * median_dist, 0.5)
            kernel_hyper_KSD = {
                'bandwidth': bandwidth
            }
            KSD, KSD_bootstrap = bootstrap_KSD(n_boots, p_x, q_samples.clone().requires_grad_(),
                                               SE_kernel_multi, trace_SE_kernel_multi,
                                               kernel_hyper=kernel_hyper_KSD, split=1
                                               )
            prop_KSD = torch.mean((KSD_bootstrap >= KSD).float().type(dtype))
            time_KSD = time.time() - start_time
            time_KSD_avg += time_KSD
            KSD_value[idx_trial] = KSD.cpu().data.numpy()

            if prop_KSD < significance:
                # rejection
                KSD_correct[idx_trial] = 1
            else:
                KSD_correct[idx_trial] = 0
            print(
                'Dx:%s Per:%s trial:%s KSD:%s KSD_v:%s prop_KSD:%s time_KSD:%s' % (
                    dx, p_level, idx_trial, KSD_correct[0:idx_trial + 1].sum() / (idx_trial + 1.),
                    KSD.cpu().data.numpy(),
                    prop_KSD.cpu().data.numpy(),
                    time_KSD_avg / (idx_trial + 1.)))
            result_dict['result']['p_level:%s' % (p_level)]['time_KSD_avg'] = time_KSD_avg / (idx_trial + 1.)

        elif method == 'MMD':
            ########### MMD Test #############################
            agg_sample = torch.cat((q_samples, p_samples), dim=0)
            if agg_sample.shape[0] > 500:
                idx_crop = 500
            else:
                idx_crop = agg_sample.shape[0]
            median_dist = median_heruistic(agg_sample[0:idx_crop, :], agg_sample[0:idx_crop, :].clone())
            bandwidth = 2 * torch.pow(0.5 * median_dist, 0.5)
            kernel_hyper_MMD = {
                'bandwidth': bandwidth
            }
            MMD, MMD_bootstrap = bootstrap_MMD(n_boots, p_samples, q_samples, SE_kernel_multi,
                                               kernel_hyper=kernel_hyper_MMD, split=1
                                               )
            prop_MMD = torch.mean((MMD_bootstrap >= MMD).float().type(dtype))
            MMD_value[idx_trial] = MMD.cpu().data.numpy()

            if prop_MMD < significance:
                # rejection
                MMD_correct[idx_trial] = 1.
            else:
                MMD_correct[idx_trial] = 0.
        else:
            raise NotImplementedError('KSD, SKSD or MMD')

    # Finished the trials
    SKSD_test = np.sum(SKSD_correct) / n_trials
    KSD_test = np.sum(KSD_correct) / n_trials
    MMD_test = np.sum(MMD_correct) / n_trials

    SKSD_v = np.mean(SKSD_value, axis=-1)
    KSD_v = np.mean(KSD_value, axis=-1)
    MMD_v = np.mean(MMD_value, axis=-1)

    print('DX:%s Per:%s SKSD:%s KSD:%s MMD:%s SKSD_value:%s KSD_value:%s MMD_value:%s' % (
        dx, p_level, SKSD_test, KSD_test, MMD_test, SKSD_v, KSD_v, MMD_v))
    SKSD_results = SKSD_correct
    KSD_results = KSD_correct
    MMD_results = MMD_correct
    if method == 'SKSD':
        result_dict['result'][
            'p_level:%s' % (p_level)]['SKSD_results'] = SKSD_results

        result_dict['result'][
            'p_level:%s' % (p_level)]['SKSD_value'] = SKSD_value
    elif method == 'KSD':
        result_dict['result'][
            'p_level:%s' % (p_level)]['KSD_results'] = KSD_results
        result_dict['result'][
            'p_level:%s' % (p_level)]['KSD_value'] = KSD_value
    elif method == 'MMD':
        result_dict['result'][
            'p_level:%s' % (p_level)]['MMD_results'] = MMD_results
        result_dict['result'][
            'p_level:%s' % (p_level)]['MMD_value'] = MMD_value

    path_name = path + 'Experiments/Results/RBM_Optimal_Init/OptimizeR/RBM_Method_%s.p' % (
        method)

    with open(path_name, 'wb') as fp:
        pickle.dump(result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

























