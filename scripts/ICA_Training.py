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
import pickle
import random

path='...' # Change to your own path

parser=argparse.ArgumentParser(description='ICA Test')

parser.add_argument('--n_updates',type=int,default=15000) # training iterations
parser.add_argument('--batch_size',type=int,default=100) # batch size
parser.add_argument('--n_g_updates',type=int,default=1) # how many iterations for updating g or test function (LSD) per each iteration
parser.add_argument('--result_interval',type=int,default=1) # intervals for storing results
parser.add_argument('--flag_U', action='store_true') # Use this to activate the active slice method for initialization
parser.add_argument('--dim',type=int,default=150) # dimensions for the ICA
parser.add_argument('--method',type=str,default='SKSD') # KSD and LSD
parser.add_argument('--optimal_init', action='store_true') # Use this to activate the active slice method for initialization
parser.add_argument('--method_score_q',type=str,default='kernel_smooth') # gradient_estimator
parser.add_argument('--flag_opt_r', action='store_true') # Use this to activate the active slice method for initialization
parser.add_argument('--disable_gpu',action='store_true') # set this to disable gpu
parser.add_argument('--base_distribution',type=str, default='multi-t') # Define the base distribution. Note that active slice not work for Laplace base distribution.
parser.add_argument('--optimal_interval1',type=int,default=200) # At early stage training (iterations<6000), how many iteration interval we update r,g with active slice
parser.add_argument('--optimal_interval2',type=int,default=200) # At later stage training (iterations>6000), how many iteration interval we update r,g with active slice



args=parser.parse_args()


dtype = torch.FloatTensor
if not args.disable_gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    dtype = torch.cuda.FloatTensor

torch.set_default_tensor_type(dtype)


def ICA_un_log_likelihood_helper(b, W):
    return lambda samples: ICA_un_log_likelihood(samples, b, W)

# hyperparameters
flag_enable_optimal = args.optimal_init
flag_U=args.flag_U
flag_opt_r = args.flag_opt_r
n_update = args.n_updates
base_distribution = args.base_distribution
eps = 0.001
n_test_update = args.n_g_updates  # iterations for updating g or test function(LSD)
batch_size = args.batch_size  # batch size
test_size = 5000  # test dataset size
train_size = 20000  # training dataset size

optimal_interval1=args.optimal_interval1
optimal_interval2=args.optimal_interval2

method_score_q=args.method_score_q
method=args.method
random_seed_list=[60,70,80,90,100]
dim = args.dim

# Store results
result_dict={}

print('Dimensions:%s' % (dim))


length_seed = len(random_seed_list)
len_tr = int(n_update / float(args.result_interval))  # compute length needed for storing results

# matrix for storing NLL results
NLL_LSD = np.zeros((length_seed, len_tr))
NLL_KSD = np.zeros((length_seed, len_tr))
NLL_maxSKSD = np.zeros((length_seed, len_tr))

# Store value
Value_LSD = np.zeros((length_seed, len_tr))
Value_KSD = np.zeros((length_seed, len_tr))
Value_maxSKSD = np.zeros((length_seed, len_tr))

# iterate through random seed


for counter_seed,seed in enumerate(random_seed_list):
    print('seed number:%s' % (counter_seed))
    print('Current running: dim %s method %s enable_optimal %s method_q %s opt_r %s' % (
    dim, method, flag_enable_optimal, method_score_q, flag_opt_r))

    # Fix random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Define base distribution
    # Define Laplace
    base_mean = torch.zeros(dim)
    base_scale = torch.ones(dim)
    if base_distribution == 'Laplace':
        base_dist = multivariate_Laplace(base_mean, base_scale)
    elif base_distribution == 'multi-t':
        base_dist = multivariate_t(base_mean, 5, base_scale)
    else:
        raise NotImplementedError('Base distribution should be either Laplace or multi-t')

    # Generate ground-truth weight matrix W
    W_true = ICA_generate_random_W_matrix(dim, scale=1., flag_from_inverse=False,dtype=dtype)  # generate ground truth W matrix
    W_true = torch.inverse(W_true)  # get inverse, use inverse as the weight matrix for training stability W=w'^{-1}
    W_inv_global = ICA_generate_random_W_matrix(dim,dtype=dtype)  # generate initialized W for model ICA

    # Generate test and training data
    test_data = ICA_generate_data(W_true, test_size, base_dist)  # generate test data using ground truth W

    train_data = ICA_generate_data(W_true, train_size, base_dist)  # generate training data using ground truth W
    # Dataloader for train
    train_dataset = Sample_Dataset(train_data)  # Create dataset for the training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Create dataloader for training

    # Define MLP for test function in LSD
    # Define test function (Neural network) config (For LSD)
    hidden_layer_num = 2  # hidden layer number
    hidden_unit_test = [300, 300]  # Hidden units
    activations_output = None  # Activation function at output layer, None for disabling the output activations
    flag_only_output_layer = False  # True for zero hidden layers (normally set to False)
    test_func = fc_Encoder_Decoder(input_dim=dim, output_dim=dim, hidden_layer_num=hidden_layer_num,
                                   hidden_unit=hidden_unit_test, activations='Swish',
                                   activations_output=activations_output,
                                   flag_only_output_layer=flag_only_output_layer
                                   )
    Adam_test = torch.optim.Adam(list(test_func.parameters()), lr=eps,
                                 betas=(0.5, 0.9))  # optimizer for test functions of LSD

    p_bar = tqdm(total=n_update)
    ###### Start Training
    if method == 'LSD':
        W_inv = W_inv_global.clone().detach().requires_grad_()  # initialized model parameters W
        Adam_W_inv = torch.optim.Adam([W_inv], lr=eps, betas=(0.9, 0.99))  # optimizers for model parameters W
        # counters
        counter_ep = 0
        counter_step = 0 # counter total iteration number
        counter_epoch = 0 # counter equivalent epoch number
        print('Run LSD')
        while counter_step < n_update:  # running fixed enumer of iterations
            for i, x in enumerate(train_loader):  # enumerate train data
                # update test_func by maximizing SD with NN
                for j in range(n_test_update):
                    Adam_test.zero_grad()
                    x = x.cuda().requires_grad_()
                    un_log_likelihood = ICA_un_log_likelihood(x, base_dist, W_inv)  # compute un-normalized LL for ICA
                    score = torch.autograd.grad(un_log_likelihood.sum(), x, create_graph=True)[
                        0]  # N x D # compute score \nabla_x log p(x)
                    div, div_reg = compute_SD(x, test_func, None, m=10, lam=1, score=score)
                    (-div_reg).backward()
                    Adam_test.step()
                # Update model parameter by minimizing LSD
                Adam_W_inv.zero_grad()
                un_log_likelihood = ICA_un_log_likelihood(x, base_dist, W_inv)
                score = torch.autograd.grad(un_log_likelihood.sum(), x, create_graph=True)[0]  # N x D
                div, div_reg = compute_SD(x, test_func, None, m=10, lam=1, score=score)
                (div).backward()

                Adam_W_inv.step()

                counter_step += 1
                p_bar.update(1)

                # Evaluation, compute test NLL for ICA
                if (counter_step) % args.result_interval == 0:
                    if counter_ep >= len_tr:
                        counter_ep = int(len_tr - 1)
                    mean_log = ICA_log_likelihood(test_data, base_dist, W_inv=W_inv.clone().detach(), flag_no_grad=True)
                    mean_test_un = ICA_un_log_likelihood(test_data, base_dist, W_inv.clone().detach())
                    NLL_LSD[counter_seed, counter_ep] = mean_log.cpu().data.numpy()
                    Value_LSD[counter_seed, counter_ep] = div.cpu().data.numpy()
                    counter_ep += 1

            counter_epoch += 1
            if counter_epoch % 1 == 0:
                print('LSD step:%s mean_test_log:%s mean_test_un:%s' % (
                    counter_step, mean_log.cpu().data.numpy(), mean_test_un.mean().cpu().data.numpy()))
        p_bar.close()
        print('LSD step:%s mean_test_log:%s mean_test_un:%s' % (
            counter_step, mean_log.cpu().data.numpy(), mean_test_un.mean().cpu().data.numpy()))

    elif method == 'SKSD':
        W_inv = W_inv_global.clone().detach().requires_grad_()  # set initialized model parameter
        Adam_W_inv = torch.optim.Adam([W_inv], lr=eps, betas=(0.9, 0.99))  # optimizer for model parameters
        g = torch.eye(dim).requires_grad_()  # initialized g matrix
        r = torch.eye(dim).requires_grad_()  # initialized r matrix
        Adam_g = torch.optim.Adam([g], lr=eps, betas=(0.9, 0.99))  # optimizer for g
        Adam_r = torch.optim.Adam([r], lr=eps, betas=(
            0.9, 0.99))  # optimizer for r (may not needed later depends on whether you want to optimize r or not)

        n_g_update = args.n_g_updates  # update iterations for g
        counter_ep = 0
        counter_step = 0
        counter_epoch = 0
        # kernel parameters for maxSVGD (None for median heuristic)
        kernel_hyper_maxSVGD = {
            'bandwidth': None
        }
        print('Run maxSKSD')

        # Start training
        while counter_step < n_update:
            for i, x in enumerate(train_loader):

                if flag_enable_optimal:  # enable r and g optimization with active slice
                    # How many iteration interval we use active slice method to update r,g.
                    if counter_step > 6000:  # 6000
                        optimal_interval = optimal_interval2  # for exact, 3000 seems to work well
                    else:
                        optimal_interval = optimal_interval1  # for exact, 1000 seems to work well

                    if counter_step % optimal_interval == 0:  # Update r and g with active slice

                        if method_score_q == 'gradient_estimator' or dim > 100:
                            idx_shuffle = torch.randperm(train_data.shape[0])[0:2000]  # use 2000 training samples if GE, otherwise CUDA out of memory
                        else:
                            idx_shuffle = torch.randperm(train_data.shape[0])[0:3000]  # use 3000 training samples for others

                        tr_sample = train_data[idx_shuffle, :].clone().detach()

                        ICA_un_log_likelihood_p_helper = ICA_un_log_likelihood_helper(base_dist, W_inv)
                        ICA_un_log_likelihood_true_helper = ICA_un_log_likelihood_helper(base_dist, W_true.inverse())

                        if method_score_q == 'kernel_smooth':

                            r = r_kernel(tr_sample, tr_sample, ICA_un_log_likelihood_p_helper, threshold=None,
                                         num_selection=dim, fix_sample=True)

                            r = r.clone().detach().requires_grad_()

                            g = Poincare_g_kernel_SVD(tr_sample, tr_sample, ICA_un_log_likelihood_p_helper, r=r,
                                                      fix_sample=True,
                                                      einsum=True)
                            g = g.clone().detach().requires_grad_()
                        elif method_score_q == 'gradient_estimator':
                            tr_sample.requires_grad_()
                            with torch.no_grad():
                                median_dist = median_heruistic(tr_sample, tr_sample.clone())
                                bandwidth = 2 * torch.pow(0.5 * median_dist, 0.5)
                            score_q_esti = gradient_estimate_im(tr_sample, bandwidth, lam=0.01)
                            r = r_optimal(tr_sample, ICA_un_log_likelihood_p_helper, None, threshold=None,
                                          num_selection=dim,
                                          score_q=score_q_esti)
                            r = r.clone().detach().requires_grad_()

                            g = Poincare_g_optimal_SVD(tr_sample, ICA_un_log_likelihood_p_helper, None, r=r,
                                                       score_q=score_q_esti)

                            g = g.clone().detach().requires_grad_()
                        elif method_score_q == 'exact':
                            tr_sample.requires_grad_()

                            r = r_optimal(tr_sample, ICA_un_log_likelihood_p_helper, ICA_un_log_likelihood_true_helper,
                                          threshold=None, num_selection=dim)
                            r = r.clone().detach().requires_grad_()

                            g = Poincare_g_optimal_SVD(tr_sample, ICA_un_log_likelihood_p_helper,
                                                       ICA_un_log_likelihood_true_helper, r=r)

                            g = g.clone().detach().requires_grad_()


                        else:
                            raise NotImplementedError


                        Adam_g = torch.optim.Adam([g], lr=eps, betas=(0.9, 0.99))  # optimizer for g
                        Adam_r = torch.optim.Adam([r], lr=eps, betas=(
                            0.9, 0.99))  # optimizer for r (may not needed later depends on whet

                # use GO to update r,g
                for j in range(n_g_update):  # update g and/or r
                    Adam_r.zero_grad()
                    Adam_g.zero_grad()
                    x = x.cuda().requires_grad_()
                    samples1 = x.clone().detach().requires_grad_()
                    samples2 = samples1.clone().detach().requires_grad_()  # clone samples
                    # compute un-normalized log likelihood
                    un_log_likelihood1 = ICA_un_log_likelihood(samples1, base_dist, W_inv)
                    score1 = torch.autograd.grad(un_log_likelihood1.sum(), samples1, create_graph=True)[0]
                    un_log_likelihood2 = ICA_un_log_likelihood(samples2, base_dist, W_inv)
                    score2 = torch.autograd.grad(un_log_likelihood2.sum(), samples2, create_graph=True)[0]
                    # normalize the g and r
                    g_n = g / (torch.norm(g, 2, dim=-1, keepdim=True) + 1e-10)
                    r_n = r / (torch.norm(r, 2, dim=-1, keepdim=True) + 1e-10)

                    # compute maxSKSD
                    maxSSD, _ = compute_max_DSSD_eff(samples1, samples2, None,
                                                     SE_kernel, d_SE_kernel, dd_SE_kernel,
                                                     flag_U=flag_U, kernel_hyper=kernel_hyper_maxSVGD,
                                                     r=r_n, g=g_n, score_samples1=score1,
                                                     score_samples2=score2, flag_median=True, median_power=0.5,
                                                     bandwidth_scale=1, scale_diag=1
                                                     )

                    (-maxSSD).backward()
                    Adam_g.step()
                    if flag_opt_r:
                        Adam_r.step()

                ## Update model by minimizing SKSD with r and updated g
                Adam_W_inv.zero_grad()
                samples1 = x.clone().detach().requires_grad_()
                samples2 = samples1.clone().detach().requires_grad_()
                un_log_likelihood1 = ICA_un_log_likelihood(samples1, base_dist, W_inv)
                score1 = torch.autograd.grad(un_log_likelihood1.sum(), samples1, create_graph=True)[0]
                un_log_likelihood2 = ICA_un_log_likelihood(samples2, base_dist, W_inv)
                score2 = torch.autograd.grad(un_log_likelihood2.sum(), samples2, create_graph=True)[0]

                g_n = g / (torch.norm(g, 2, dim=-1, keepdim=True) + 1e-10)
                r_n = r / (torch.norm(r, 2, dim=-1, keepdim=True) + 1e-10)
                maxSSD, _ = compute_max_DSSD_eff(samples1, samples2, None,
                                                 SE_kernel, d_SE_kernel, dd_SE_kernel,
                                                 flag_U=flag_U, kernel_hyper=kernel_hyper_maxSVGD,
                                                 r=r_n, g=g_n, score_samples1=score1,
                                                 score_samples2=score2, flag_median=True, median_power=0.5,
                                                 bandwidth_scale=1, scale_diag=1
                                                 )

                (maxSSD).backward()
                Adam_W_inv.step()
                p_bar.update(1)
                counter_step += 1

                # Evaluation
                if (counter_step) % args.result_interval == 0:
                    if counter_ep >= len_tr:
                        counter_ep = int(len_tr - 1)
                    mean_log = ICA_log_likelihood(test_data, base_dist, W_inv=W_inv.clone().detach(),
                                                  flag_no_grad=True)
                    mean_test_un = ICA_un_log_likelihood(test_data, base_dist, W_inv.clone().detach())

                    NLL_maxSKSD[counter_seed, counter_ep] = mean_log.cpu().data.numpy()
                    Value_maxSKSD[counter_seed, counter_ep] = maxSSD.cpu().data.numpy()

                    counter_ep += 1

            counter_epoch += 1
            if counter_epoch % 1 == 0:
                print('maxSKSD step:%s mean_test_log:%s mean_test_un:%s maxSKSD_value:%s' % (
                    counter_step, mean_log.cpu().data.numpy(), torch.mean(mean_test_un).cpu().data.numpy(),
                    maxSSD.cpu().data.numpy()))
        p_bar.close()
        print('maxSKSD step:%s mean_test_log:%s mean_test_un:%s' % (
            counter_step, mean_log.cpu().data.numpy(), torch.mean(mean_test_un).cpu().data.numpy()))



    elif method == 'KSD':
        ########################## KSD ####################################
        W_inv = W_inv_global.clone().detach().requires_grad_()
        Adam_W_inv = torch.optim.Adam([W_inv], lr=eps, betas=(0.9, 0.99))
        counter_ep = 0
        counter_step = 0
        counter_epoch = 0
        print('Run KSD')  ### For KSD,must set flag U to be True
        while counter_step < n_update:
            for i, x in enumerate(train_loader):
                Adam_W_inv.zero_grad()
                x = x.cuda().requires_grad_()
                samples1 = x.clone().detach().requires_grad_()
                samples2 = samples1.clone().detach().requires_grad_()
                un_log_likelihood1 = ICA_un_log_likelihood(samples1, base_dist, W_inv)
                score1 = torch.autograd.grad(un_log_likelihood1.sum(), samples1, create_graph=True)[0]
                un_log_likelihood2 = ICA_un_log_likelihood(samples2, base_dist, W_inv)
                score2 = torch.autograd.grad(un_log_likelihood2.sum(), samples2, create_graph=True)[0]

                median_dist = median_heruistic(x, x.clone())  # compute median heuristic

                bandwidth = 2 * np.sqrt(1. / (2 * np.log(batch_size))) * torch.pow(0.5 * median_dist, 0.5)
                kernel_hyper_KSD = {
                    'bandwidth': bandwidth
                }

                # update model parameters by minimizing KSD
                KSD, _ = compute_KSD(samples1, samples2, None, SE_kernel_multi, trace_SE_kernel_multi, flag_U=True,
                                     kernel_hyper=kernel_hyper_KSD, score_sample1=score1, score_sample2=score2,
                                     flag_retain=True, flag_create=False
                                     )  # flag_retain is for back-propogation, because we need to back-propagate through KSD
                (KSD).backward()
                Adam_W_inv.step()
                p_bar.update(1)
                counter_step += 1

                # Evaluation
                if (counter_step) % args.result_interval == 0:
                    if counter_ep >= len_tr:
                        counter_ep = int(len_tr - 1)
                    mean_log = ICA_log_likelihood(test_data, base_dist, W_inv=W_inv.clone().detach(), W=None,
                                                  flag_no_grad=True)
                    mean_test_un = ICA_un_log_likelihood(test_data, base_dist, W_inv.clone().detach())

                    NLL_KSD[counter_seed, counter_ep] = mean_log.cpu().data.numpy()
                    Value_KSD[counter_seed, counter_ep] = KSD.cpu().data.numpy()

                    counter_ep += 1

            counter_epoch += 1
            if counter_epoch % 1 == 0:
                print('KSD step:%s KSD:%s mean_test_log:%s mean_test_un:%s' % (
                    counter_step, KSD.cpu().data.numpy(), mean_log.cpu().data.numpy(),
                    torch.mean(mean_test_un).cpu().data.numpy()))
        p_bar.close()
        print('KSD step:%s KSD:%s mean_test_log:%s mean_test_un:%s' % (
            counter_step, KSD.cpu().data.numpy(), mean_log.cpu().data.numpy(),
            torch.mean(mean_test_un).cpu().data.numpy()))

    # Store results
    if method == 'LSD':
        result_dict['NLL'] = NLL_LSD
        result_dict['Value'] = Value_LSD
        name = 'ICA_%s_dim%s.p' % (method, dim)
    elif method == 'SKSD':
        result_dict['NLL'] = NLL_maxSKSD
        result_dict['Value'] = Value_maxSKSD
        name = 'ICA_%s_OptimalInit%s_%s_Optr%s_dim%s.p' % (
        method, flag_enable_optimal, method_score_q, flag_opt_r, dim)
    elif method == 'KSD':
        result_dict['NLL'] = NLL_KSD
        result_dict['Value'] = Value_KSD
        name = 'ICA_%s_dim%s.p' % (method, dim)
    path_name = path + name
    with open(path_name, 'wb') as fp:
        pickle.dump(result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)










