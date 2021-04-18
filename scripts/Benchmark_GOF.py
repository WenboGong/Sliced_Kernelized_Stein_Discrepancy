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
from src.Network import *
from src.Divergence import *
from src.Kernel import *
from src.distribution import *
from src.GOF_Test import *
from src.active_slice import *
from src.Dataloader import *
import time
import pickle

# Arguments for scripts
parser=argparse.ArgumentParser(description='Benchmark GOF test')
parser.add_argument('--n_trials',type=int,default=100) # number of GOF trials
parser.add_argument('--n_samples',type=int,default=1000) # number of total sample used
parser.add_argument('--n_tr',type=int,default=200) # number of training samples
parser.add_argument('--grad_opt_ep',type=int,default=50) # number of GO steps for slice directions
parser.add_argument('--num_select',type=int,default=-1) # pruning level: how many slice directions are chosen, -1 means slice number = dimensions
parser.add_argument('--batch_size',type=int,default=50) # batch size for GO slice training
parser.add_argument('--method',type=str,default='SKSD') # GOF test method, either KSD,SKSD or MMD
parser.add_argument('--distribution',type=str,default='Laplace') # test distributions: Laplace, diffusion, multi-t or Null
parser.add_argument('--method_score_q',type=str,default='exact') # score estimation method for active slice: kernel_smooth or gradient_estimator or exact
parser.add_argument('--method_r_opt',type=str,default='Fixed') # Whether manually fixed r or optimal r: Fixed or Not. For benchmark, use Fixed.
parser.add_argument('--optimal_init', action='store_true') # Use this to activate the active slice method for initialization
parser.add_argument('--optimal_grad', action='store_true') # Use this to activate GO for slice direction.
# Note at least use one of them. E.g. set --optimal_init to enable just active slice, set --optimal_grad to use just GO, and set both to use active slice for initialization and then GO refinement.
parser.add_argument('--initialization',type=str,default='eye') # initialization method for slice r and g: eye means identity initialization, randn means Gaussian random initialization.
parser.add_argument('--disable_gpu',action='store_true') # set this to disable gpu

args=parser.parse_args()

path='...' # Set this to your project working directory


if not args.disable_gpu:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Set random seed
torch.manual_seed(0)
np.random.seed(0)

# hyperparameters
n_trials=args.n_trials
n_samples=args.n_samples
n_tr=args.n_tr
significance=0.05 # significance level for GOF
n_boots=1000 # number of bootstrap samples, cause large GPU memory usage. Reduce this if CUDA out of memory.
num_select=args.num_select
batch_size=args.batch_size
grad_opt_ep=args.grad_opt_ep

method=args.method # KSD or MMD or SKSD
distribution=args.distribution
method_score_q=args.method_score_q
method_r_opt=args.method_r_opt

flag_optimal_init = args.optimal_init
flag_optimal_grad = args.optimal_grad


if not flag_optimal_grad and not flag_optimal_init:
    warnings.warn("Set at least --optimal_init or --optimal_grad. Otherwise, r and g only uses initialized values.")

dim_array=np.unique(np.around(np.linspace(2,100,num=15))) # array of dimensions for the benchmark problems.
bandwidth_Power_list=[0.5] # bandwidth for median heuristic, use 0.5 = sqrt(median distance)

# Structure for storing results
result_dict={}

result_dict=para_store(result_dict,
n_trials=args.n_trials,
n_samples=args.n_samples,
n_tr=args.n_tr,
significance=significance,
n_boots=n_boots,
num_select=args.num_select,
batch_size=args.batch_size,
grad_opt_ep=args.grad_opt_ep,
Kernel_Choice='RBF',
method=args.method, # KSD or MMD
distribution=args.distribution, # or multi-t
method_score_q=args.method_score_q, # gradient_estimator
method_r_opt=args.method_r_opt,
dim_array=dim_array,
initialization=args.initialization,
                       flag_optimal_grad=flag_optimal_grad,
                       flag_optimal_init=flag_optimal_init
                       )


result_dict['result']={}

print('Bandwidth Power:0.5')


# Iterate through dimensions
for counter_dim,dim in enumerate(dim_array):
    if args.num_select == -1:
        # ==-1 means no pruning
        num_select = int(dim)

    # Print experiment setup
    print('Dim:%s name:%s' % (dim, '%s_%s_%s_%s_NumSelect%s_OI%s_OG%s_Init%s.p' % (
        method, distribution, method_r_opt, method_score_q, args.num_select, flag_optimal_init, flag_optimal_grad,
        args.initialization)))

    # For storing results
    SKSD_results = np.zeros((n_trials))
    KSD_results = np.zeros((n_trials))
    MMD_results = np.zeros((n_trials))

    KSD_value = np.zeros((n_trials))
    SKSD_value = np.zeros((n_trials))
    MMD_value = np.zeros((n_trials))
    # For storing time usage
    time_optimal_init = 0
    time_grad = 0
    time_SKSD_avg = 0
    time_KSD_avg = 0
    # For storing Correct rejection
    SKSD_correct = np.zeros(n_trials)
    KSD_correct = np.zeros(n_trials)
    MMD_correct = np.zeros(n_trials)

    dim = int(dim)
    result_dict['result']['dim:%s' % (dim)] = {}

    # Set distribution p
    p_cov = torch.eye(dim)
    p_mean = torch.zeros(dim)
    sample_size_full = torch.Size([n_samples]) # total sample number
    p_x = torch.distributions.multivariate_normal.MultivariateNormal(p_mean, p_cov)
    # Set distribution q
    if distribution == 'Laplace':
        sample_size = torch.Size([n_samples])
        q_cov = 1. / np.sqrt(2) * torch.ones(dim)
        H_1_q = multivariate_Laplace(p_mean, q_cov)

    elif distribution == 'multi-t':

        p_cov = p_cov * 5 / (5 - 2.0)
        sample_size = torch.Size([n_samples, dim])
        H_1_q = multivariate_t(df=5, loc=0)
    elif distribution == 'diffusion':
        q_cov = p_cov.clone().detach()
        q_cov[0, 0] = 0.3
        sample_size = torch.Size([n_samples])
        H_1_q = torch.distributions.multivariate_normal.MultivariateNormal(p_mean, q_cov)
    elif distribution == 'Null':
        sample_size = torch.Size([n_samples])
        H_1_q = torch.distributions.multivariate_normal.MultivariateNormal(p_mean, p_cov)

    # Start the trials
    for idx_trial in tqdm(range(n_trials)):
        # Initialize r,g
        if args.initialization == 'eye':
            r = torch.eye(dim)[0:num_select, :]
            g = torch.randn(dim, dim)[0:num_select, :] # Has to be randn in this benchmark problem for g, because eye initialization coincides with the global optimal g.
        elif args.initialization == 'randn':

            r = torch.randn(dim, dim)[0:num_select, :]
            g = torch.randn(dim, dim)[0:num_select, :]
        else:
            raise NotImplementedError

        # Draw samples
        q_samples = H_1_q.rsample(sample_size_full).requires_grad_()

        test_sample = q_samples.clone().detach().requires_grad_()
        p_samples = p_x.rsample(sample_size_full)

        # Split samples into training and test set
        if flag_optimal_init or flag_optimal_grad:

            ind = torch.randperm(q_samples.shape[0])
            ind_tr = ind[0:n_tr]
            ind_test = ind[n_tr:]
            tr_sample = q_samples[ind_tr, :].clone().detach().requires_grad_()
            test_sample = q_samples[ind_test, :].clone().detach().requires_grad_()

        # Active Slice method for initialization
        if flag_optimal_init:
            if method_score_q == 'gradient_estimator': # GE active slice
                time_start = time.time()
                with torch.no_grad():
                    median_dist = median_heruistic(tr_sample, tr_sample.clone())
                    bandwidth = 2 * torch.pow(0.5 * median_dist, 0.5) # sqrt(med_distance)
                score_q_esti = gradient_estimate_im(tr_sample, bandwidth, lam=0.01) # Stein gradient estimator by Li.
                if method_r_opt != 'Fixed': # use active slice method to find r.
                    raise NotImplementedError('For benchmark problem, fixed r should be used. Because active slice for r is not stable in this case due to the robustness issue of noisy eigenvalue decomposition.')
                g = Poincare_g_optimal_SVD(tr_sample, p_x.log_prob, None, r=r, score_q=score_q_esti)
                time_optimal_init = time.time() - time_start
            elif method_score_q == 'kernel_smooth':
                time_start = time.time()
                # tr_sample=H_1_q.rsample(torch.Size([500])).requires_grad_()
                if method_r_opt != 'Fixed':
                    raise NotImplementedError('For benchmark problem, fixed r should be used. Because active slice for r is not stable in this case due to the robustness issue of noisy eigenvalue decomposition.')

                g = Poincare_g_kernel_SVD(tr_sample, tr_sample, p_x.log_prob, r=r, fix_sample=True,
                                          lobpcg=False)
                time_optimal_init = time.time() - time_start
            elif method_score_q == 'exact':
                time_start = time.time()
                if method_r_opt != 'Fixed':
                    raise NotImplementedError('For benchmark problem, fixed r should be used. Because active slice for r is not stable in this case due to the robustness issue of noisy eigenvalue decomposition.')
                g = Poincare_g_optimal_SVD(tr_sample, p_x.log_prob, H_1_q.log_prob, r=r) # with exact score_q
                time_optimal_init = time.time() - time_start

        # GO for slice direction
        if flag_optimal_grad:
            # training samples
            tr_sample.requires_grad_()
            # optimizer for r and g
            r_init = r.clone().detach().requires_grad_()
            Adam_r = torch.optim.Adam([r_init], lr=0.001, betas=(0.9, 0.99))

            r = r_init
            g_init = g.clone().detach().requires_grad_()

            Adam_g = torch.optim.Adam([g_init], lr=0.001, betas=(0.9, 0.99))

            # normalize g to g_n
            g = g_init / (torch.norm(g_init, 2, dim=-1, keepdim=True) + 1e-10)
            # Define data loader for training data
            tr_dataset = Sample_Dataset(tr_sample.clone().detach())
            loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)

            time_start_grad = time.time()
            # Start GO for slice

            for i in range(grad_opt_ep):
                for j, data in enumerate(loader):
                    Adam_r.zero_grad()
                    Adam_g.zero_grad()
                    kernel_hyper_maxSVGD = {
                        'bandwidth': None
                    }
                    data.requires_grad_()
                    start_time = time.time()
                    # Compute score_p
                    score1 = torch.autograd.grad(p_x.log_prob(data).sum(), data)[0]
                    # Compute maxSKSD
                    diver, _ = compute_max_DSSD_eff(data, data.clone().detach(), None, SE_kernel,
                                                    d_kernel=d_SE_kernel,
                                                    dd_kernel=dd_SE_kernel,
                                                    r=r, g=g, kernel_hyper=kernel_hyper_maxSVGD,
                                                    score_samples1=score1,
                                                    score_samples2=score1.clone().detach()
                                                    , flag_median=True, flag_U=False, median_power=0.5,
                                                    bandwidth_scale=1
                                                    )

                    (-diver).backward()
                    if method_r_opt != 'Fixed':
                        # if use fixed r, do not update it.
                        Adam_r.step()

                    Adam_g.step()
                    # re-normalize to unit vectors.
                    r = r_init / (torch.norm(r_init, 2, dim=-1, keepdim=True) + 1e-10)
                    g = g_init / (torch.norm(g_init, 2, dim=-1, keepdim=True) + 1e-10)

            time_grad = time.time() - time_start_grad

        # re-normalize to unit vector
        g = g.clone().detach() / (torch.norm(g, 2, dim=-1, keepdim=True) + 1e-10)
        r = r.clone().detach() / (torch.norm(r, 2, dim=-1, keepdim=True) + 1e-10)

        # Start GOF test
        if method == 'SKSD':
            kernel_hyper_SKSD = {
                'bandwidth': None
            }
            ############ SKSD Test ########################
            time_start = time.time()
            SKSD, SKSD_bootstrap = bootstrap_SKSD(n_boots, p_x, test_sample, r, g,
                                                  SE_kernel, kernel_hyper=kernel_hyper_SKSD,
                                                  median_power=0.5, d_kernel=d_SE_kernel,
                                                  dd_kernel=dd_SE_kernel
                                                  )
            prop_SKSD = torch.mean((SKSD_bootstrap >= SKSD).float().cuda()) # proportion that bootstrap >= SKSD value
            # total time of the test for 1 trial
            time_SKSD = time.time() - time_start + time_optimal_init + time_grad
            time_SKSD_avg += time_SKSD
            # Store SKSD value
            SKSD_value[idx_trial] = SKSD.cpu().data.numpy()
            if prop_SKSD < significance:
                # reject alternative
                SKSD_correct[idx_trial] = 1.
            else:
                SKSD_correct[idx_trial] = 0.
            if (idx_trial + 1) % 1 == 0:
                print('trial:%s test_acc:%s prop_SKSD:%s SKSD:%s time:%s' % (
                    idx_trial, SKSD_correct[0:idx_trial + 1].sum() / (idx_trial + 1), prop_SKSD,
                    SKSD.cpu().data.numpy(), time_SKSD_avg / (idx_trial + 1)))

        elif method == 'KSD':
            ############ KSD Test ##########################
            start_time = time.time()
            median_dist = median_heruistic(q_samples, q_samples.clone())
            bandwidth = 2 * torch.pow(0.5 * median_dist, 0.5)
            kernel_hyper_KSD = {
                'bandwidth': bandwidth
            }
            KSD, KSD_bootstrap = bootstrap_KSD(n_boots, p_x, q_samples.clone().requires_grad_(),
                                               SE_kernel_multi, trace_SE_kernel_multi,
                                               kernel_hyper=kernel_hyper_KSD
                                               )
            prop_KSD = torch.mean((KSD_bootstrap >= KSD).float().cuda())
            time_KSD = time.time() - start_time

            time_KSD_avg += time_KSD

            KSD_value[idx_trial] = KSD.cpu().data.numpy()

            if prop_KSD < significance:
                # reject alternative
                KSD_correct[idx_trial] = 1
            else:
                KSD_correct[idx_trial] = 0
            if (idx_trial + 1) % 1 == 0:
                print('trial:%s test_acc:%s prop_KSD:%s KSD:%s time:%s' % (
                    idx_trial, KSD_correct[0:idx_trial + 1].sum() / (idx_trial + 1), prop_KSD,
                    KSD.cpu().data.numpy(), time_KSD_avg / (idx_trial + 1)))
        elif method == 'MMD':
            ########### MMD Test #############################
            agg_sample = torch.cat((q_samples, p_samples), dim=0)
            median_dist = median_heruistic(agg_sample, agg_sample.clone())
            bandwidth = 2 * torch.pow(0.5 * median_dist, 0.5)
            kernel_hyper_MMD = {
                'bandwidth': bandwidth
            }
            MMD, MMD_bootstrap = bootstrap_MMD(n_boots, p_samples, q_samples, SE_kernel_multi,
                                               kernel_hyper=kernel_hyper_MMD
                                               )
            prop_MMD = torch.mean((MMD_bootstrap >= MMD).float().cuda())
            MMD_value[idx_trial] = MMD.cpu().data.numpy()

            if prop_MMD < significance:
                # reject alternative
                MMD_correct[idx_trial] = 1.
            else:
                MMD_correct[idx_trial] = 0.
            if (idx_trial + 1) % 1 == 0:
                print('trial:%s test_acc:%s prop_MMD:%s MMD:%s' % (
                    idx_trial, MMD_correct[0:idx_trial + 1].sum() / (idx_trial + 1), prop_MMD,
                    MMD.cpu().data.numpy()))

    # Finished all trials and process the results
    SKSD_test = np.sum(SKSD_correct) / n_trials
    KSD_test = np.sum(KSD_correct) / n_trials
    MMD_test = np.sum(MMD_correct) / n_trials

    SKSD_v = np.mean(SKSD_value)
    KSD_v = np.mean(KSD_value)
    MMD_v = np.mean(MMD_value)

    SKSD_results = SKSD_correct
    KSD_results = KSD_correct
    MMD_results = MMD_correct

    if method == 'SKSD':
        result_dict['result']['dim:%s' % (dim)]['SKSD_result'] = SKSD_results
        result_dict['result']['dim:%s' % (dim)]['SKSD_value'] = SKSD_value
        result_dict['result']['dim:%s' % (dim)][
            'SKSD_time'] = time_SKSD_avg / (n_trials)


    elif method == 'KSD':
        result_dict['result']['dim:%s' % (dim)][
            'KSD_result'] = KSD_results
        result_dict['result']['dim:%s' % (dim)]['KSD_value'] = KSD_value
        result_dict['result']['dim:%s' % (dim)][
            'KSD_time'] = time_KSD_avg / (n_trials)
    elif method == 'MMD':
        result_dict['result']['dim:%s' % (dim)][
            'MMD_result'] = MMD_results
        result_dict['result']['dim:%s' % (dim)]['MMD_value'] = MMD_value

    #### Save results
    path_store = '....' # Change to your own store path

    name1 = '%s_%s_%s_%s_NumSelect%s_OI%s_OG%s_Init%s.p' % (
    method, distribution, method_r_opt, method_score_q, args.num_select, flag_optimal_init, flag_optimal_grad,
    args.initialization)
    with open(path_store + name1, 'wb') as fp:
        pickle.dump(result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)




















