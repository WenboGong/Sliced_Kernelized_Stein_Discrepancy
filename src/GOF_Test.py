import torch
import numpy as np
from src.Divergence import *
def bootstrap_SKSD(n_boots,p_x,q_samples,r,g,kernel,kernel_hyper,median_power,**kwargs):
    n_sample=q_samples.shape[0]
    weights=np.random.multinomial(n_sample,np.ones(n_sample)/n_sample,size=int(n_boots))
    weights=weights/n_sample
    weights=torch.from_numpy(weights).type(q_samples.type())

    d_kernel=kwargs['d_kernel']
    dd_kernel=kwargs['dd_kernel']
    # Compute SKSD

    SKSD, SKSD_comp = compute_max_DSSD_eff(q_samples, q_samples.clone().detach().requires_grad_(), p_x.log_prob, kernel,d_kernel,dd_kernel,flag_U=True,
                                       r=r, g=g, flag_median=True, median_power=median_power,
                                       kernel_hyper=kernel_hyper
                                       )

    with torch.no_grad():
        # now compute boots strap samples
        SKSD_comp=torch.sum(SKSD_comp,dim=0,keepdim=True) # 1 x N x N

        weights_exp=torch.unsqueeze(weights,dim=-1) # m x N x 1
        weights_exp2=torch.unsqueeze(weights,dim=1) # m x 1 x n

        bootstrap_samples=(weights_exp-1./n_sample)*SKSD_comp*(weights_exp2-1./n_sample) # m x N x N

        bootstrap_samples=torch.sum(torch.sum(bootstrap_samples,dim=-1),dim=-1)


    return SKSD,bootstrap_samples

def bootstrap_KSD(n_boots,p_x,q_samples,kernel,trace_kernel,kernel_hyper,split=1):
    n_sample = q_samples.shape[0]

    weights = np.random.multinomial(n_sample, np.ones(n_sample) / n_sample, size=int(n_boots))
    weights=weights/n_sample

    weights = torch.from_numpy(weights).type(q_samples.type())

    # Compute KSD
    KSD,KSD_comp=compute_KSD(q_samples,q_samples.clone().detach().requires_grad_(),p_x.log_prob,kernel,trace_kernel,flag_U=True,
                             kernel_hyper=kernel_hyper
                             )
    with torch.no_grad():
        # now compute boots strap samples
        KSD_comp = torch.unsqueeze(KSD_comp, dim=0)  # 1 x N x N

        weights_exp = torch.unsqueeze(weights, dim=-1)  # m x N x 1
        weights_exp2 = torch.unsqueeze(weights, dim=1)  # m x 1 x n

        bootstrap_samples = (weights_exp - 1. / n_sample) * KSD_comp * (weights_exp2 - 1. / n_sample)  # m x N x N

        bootstrap_samples = torch.sum(torch.sum(bootstrap_samples, dim=-1), dim=-1)


    return KSD,bootstrap_samples

def bootstrap_MMD(n_boots,p_samples,q_samples,kernel,kernel_hyper,split=1):
    n_sample = q_samples.shape[0]

    weights = np.random.multinomial(n_sample, np.ones(n_sample) / n_sample, size=int(n_boots))
    weights=weights/n_sample

    weights = torch.from_numpy(weights).type(q_samples.type())

    MMD,MMD_comp=compute_MMD(p_samples, q_samples, kernel, kernel_hyper, flag_U=True,flag_simple_U=True)

    # Now compute bootstrap samples
    # now compute boots strap samples
    with torch.no_grad():
        MMD_comp = torch.unsqueeze(MMD_comp, dim=0)  # 1 x N x N
        weights_exp = torch.unsqueeze(weights, dim=-1)  # m x N x 1
        weights_exp2 = torch.unsqueeze(weights, dim=1)  # m x 1 x n

        bootstrap_samples = (weights_exp - 1. / n_sample) * MMD_comp * (weights_exp2 - 1. / n_sample)  # m x N x N

        bootstrap_samples = torch.sum(torch.sum(bootstrap_samples, dim=-1), dim=-1)

    return MMD,bootstrap_samples
