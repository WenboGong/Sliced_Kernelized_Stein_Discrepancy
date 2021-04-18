import torch
from src.Kernel import *
from src.Util import *
def r_optimal(samples,score_p_func,score_q_func,threshold=30,num_selection=None,**kwargs):


    samples=samples.clone().detach()
    samples.requires_grad_()

    log_likelihood_p=score_p_func(samples) # N

    score_p=torch.autograd.grad(log_likelihood_p.sum(),samples)[0] # score function of p

    if 'score_q' in kwargs:
        # compute score of q outside
        score_q=kwargs['score_q']
    else:
        # if score_q function is available
        log_likelihood_q = score_q_func(samples)  # N
        score_q=torch.autograd.grad(log_likelihood_q.sum(),samples)[0]

    score_diff = score_p - score_q  # N x D
    score_diff_1 = score_diff.unsqueeze(-1)  # N x D x 1
    score_diff_2 = score_diff.unsqueeze(-2)  # N x 1 x D

    score_mat = torch.matmul(score_diff_1, score_diff_2)  # 1000 x D x D

    mean_score_mat = score_mat.mean(0)  # D x D


    eig_value, eig_vec = torch.symeig(mean_score_mat, eigenvectors=True) # eigenvalue-decomposition
    # Selection
    r_th = threshold # threshold for pruning. We can set it to None if num_selection is set.
    r_candidate = (eig_vec.t()).flip(0)
    eig_value_cand = eig_value.flip(0)
    max_eig_value = eig_value_cand[0] # maximized eigenvalue

    if num_selection is None:
        # num_selection is not set, then use threshold to prune
        for d in range(samples.shape[-1]):
            eig_cand = eig_value_cand[d]
            if torch.abs(max_eig_value / (eig_cand + 1e-10)) < r_th: # only select relatively large eigenvalues
                r_comp = r_candidate[d, :].unsqueeze(0)  # 1x D
                if d == 0:
                    r = r_comp
                else:
                    r = torch.cat((r, r_comp), dim=0)  # n x D
    else:
        # if num_selection is set to K, then only choose the top K eigenvectors
        for d in range(num_selection):
            r_comp = r_candidate[d, :].unsqueeze(0)  # 1x D
            if d == 0:
                r = r_comp
            else:
                r = torch.cat((r, r_comp), dim=0)  # n x D
    return r


def Poincare_g_optimal_SVD(samples,score_p_func,score_q_func,**kwargs):
    # this is to find the g that minimize the poincare inequality upper bound.
    # samples: NxD
    # score_p_func:log likelihood function
    # score_q_func: log likelihodd function
    dim=samples.shape[-1]
    if 'r' in kwargs:
        r=kwargs['r'] # r x D
    else:
        r=torch.eye(dim) # D x D


    num_r=r.shape[0] # number of r

    log_likelihood_p=score_p_func(samples) # N, likelihood of  p

    score_p=torch.autograd.grad(log_likelihood_p.sum(),samples,create_graph=True,retain_graph=True)[0] # N x D, score of p
    # score of q
    if 'score_q' in kwargs:
        score_q=kwargs['score_q']
    else:
        log_likelihood_q = score_q_func(samples)  # N
        score_q = torch.autograd.grad(log_likelihood_q.sum(), samples, create_graph=True, retain_graph=True)[0] # N x D

    # for each r, we get a corresponding g_r by first eigenvalue decomposition and then select the top 1 g as g_r.
    for d in range(num_r):
        r_candidate=r[d,:].unsqueeze(0) # 1 x D
        projected_score_diff=((score_p-score_q)*r_candidate).sum(-1) # N, projected score difference
        grad_proj_score_diff=torch.autograd.grad(projected_score_diff.sum(),samples,retain_graph=True)[0] # N x D, gradient of projected score difference
        H=grad_proj_score_diff.unsqueeze(-1)*grad_proj_score_diff.unsqueeze(-2) # N x D x D, sensitivity matrix
        H=H.mean(0) # D x D
        eig_value, eig_vec = torch.symeig(H, eigenvectors=True) # eigenvalue decomposition
        vec_candidate=eig_vec.t().flip(0)
        if d==0:
            g_comp=vec_candidate[0,:].unsqueeze(0) # 1 x D
            g=g_comp
        else:
            g_comp = vec_candidate[0, :].unsqueeze(0)  # 1 x D
            g=torch.cat((g,g_comp),dim=0) # r x D


    return g # r x D


def Poincare_g_kernel_SVD(samples1,samples2,score_p_func,**kwargs):
    # active slice with kernel smooth estimated score_q
    samples1 = samples1.clone().detach()
    samples1.requires_grad_()
    if 'lobpcg' in kwargs:
        # enable fast approximation for eigenvalue decomposition
        flag_lobpcg=kwargs['lobpcg']
    else:
        flag_lobpcg=False


    if 'fix_sample' in kwargs:
        flag_fix_sample=kwargs['fix_sample']
    else:
        flag_fix_sample=False

    if 'kernel' in kwargs:
        kernel=kwargs['kernel']
    else:
        kernel=SE_kernel_multi

    if 'r' in kwargs:
        r=kwargs['r']
    else:
        r=torch.eye(samples1.shape[-1]) # default value for r (eye initialization)

    if flag_fix_sample: # samples1=samples2

        samples2=samples1.clone().detach()
        samples2.requires_grad_()
    else:
        samples2=samples2.clone().detach()
        samples2.requires_grad_()

    num_r=r.shape[0]
    median_dist = median_heruistic(samples1[0:100, :], samples1[0:100, :].clone())
    bandwidth = 2 * torch.pow(0.5 * median_dist, 0.5)
    kernel_hyper_KSD = {
        'bandwidth': 1*bandwidth
    }
    score_p=score_p_func(samples1)# N
    score_p=torch.autograd.grad(score_p.sum(),samples1,create_graph=True,retain_graph=True)[0]# N x D


    # compute kernel matrix
    K = kernel(torch.unsqueeze(samples1, dim=1), torch.unsqueeze(samples2, dim=0),kernel_hyper=kernel_hyper_KSD) # N x N
    K_exp=K.unsqueeze(-1)
    score_p_exp = score_p.unsqueeze(-2)  # N x 1 x D
    samples1_exp = samples1.unsqueeze(-2)  # N x 1 x D
    samples2_exp = samples2.unsqueeze(-3)  # 1 x N2 x D

    for d in range(num_r):
        r_candidate = r[d, :].unsqueeze(0).unsqueeze(0)  # 1 x 1 x D


        Term1=torch.einsum('ijk,pqr,ilr->q',K_exp,r_candidate,score_p_exp)
        # only for RBF kernel. We explicit derive its derivative of this kernel,
        Term2_c1=-2./(bandwidth**2+1e-9)*torch.einsum('ijk,imr,pqr->m',K_exp,samples1_exp,r_candidate)
        Term2_c2=-2./(bandwidth**2+1e-9)*torch.einsum('ijk,mjr,pqr->m',K_exp,samples2_exp,r_candidate)
        Term2=Term2_c1-Term2_c2
        force = (Term1 + 1*Term2)  # 1
        grad_force = torch.autograd.grad(force.sum(), samples2, retain_graph=True)[0]  # sam2 x D
        H = grad_force.unsqueeze(-1) * grad_force.unsqueeze(-2)  # sam2 x D x D
        H = H.mean(0)  # D x D


        if flag_lobpcg:
            _, eig_vec_appro = torch.lobpcg(H.clone().detach(), k=1, niter=30) # approximated eigen decomposition
            if d == 0:
                g_comp = eig_vec_appro.t()  # 1 x D
                g = g_comp
            else:
                g_comp = eig_vec_appro.t()  # 1 x D
                g = torch.cat((g, g_comp), dim=0)  # r x D
        else:
            eig_value, eig_vec = torch.symeig(H.clone().detach(), eigenvectors=True) # eigenvalue decomposition
            vec_candidate = eig_vec.t().flip(0)
            if d == 0:
                g_comp = vec_candidate[0, :].unsqueeze(0)  # 1 x D
                g = g_comp
            else:
                g_comp = vec_candidate[0, :].unsqueeze(0)  # 1 x D
                g = torch.cat((g, g_comp), dim=0)  # r x D

    return g


def r_kernel(samples1,samples2,score_p_func,threshold=30,num_selection=None,**kwargs):
    # for kernel_smooth active slice
    samples1 = samples1.clone().detach()
    samples1.requires_grad_()
    if 'lobpcg' in kwargs:
        flag_lobpcg=kwargs['lobpcg']
    else:
        flag_lobpcg=False

    if 'fix_sample' in kwargs:
        flag_fix_sample = kwargs['fix_sample']
    else:
        flag_fix_sample = False

    if 'kernel' in kwargs:
        kernel = kwargs['kernel']
    else:
        kernel = SE_kernel_multi

    if flag_fix_sample:
        # sample1 =sample 2
        samples2=samples1.clone().detach()
        samples2.requires_grad_()
    else:
        samples2=samples2.clone().detach()
        samples2.requires_grad_()

    score_p=score_p_func(samples1) # N
    score_p=torch.autograd.grad(score_p.sum(),samples1)[0] # N x D

    with torch.no_grad():
        score_p = score_p.reshape((score_p.shape[0], 1, score_p.shape[1]))  # N x 1 x D
        # median distance
        median_dist = median_heruistic(samples1[0:100, :], samples1[0:100, :].clone())
        bandwidth = 2 * torch.pow(0.5 * median_dist, 0.5)
        kernel_hyper_KSD = {
            'bandwidth': 2*bandwidth
        }

        K = kernel(torch.unsqueeze(samples1, dim=1), torch.unsqueeze(samples2, dim=0)
                   , kernel_hyper=kernel_hyper_KSD)  # N x N

        Term1 = (torch.unsqueeze(K, dim=-1) * score_p)  # N x N x D

        Term2 = repulsive_SE_kernel_multi(samples1, samples2, kernel_hyper=kernel_hyper_KSD,
                                 K=torch.unsqueeze(K, dim=-1))  # N x N xD

        force = torch.mean(Term1 + 1* Term2, dim=0)  # N x D
        score_diff_1 = force.unsqueeze(-1)  # 1000 x D x 1
        score_diff_2 = force.unsqueeze(-2)  # 1000 x 1 x D

        score_mat_kernel = torch.matmul(score_diff_1, score_diff_2)
        mean_score_mat = score_mat_kernel.mean(0)  # D x D

        if flag_lobpcg==True and num_selection is not None and (num_selection*3)<mean_score_mat.shape[0]:
            # approximate eigenvectors
            eig_value_appro, eig_vec_appro = torch.lobpcg(mean_score_mat, k=num_selection, niter=10)
            r=eig_vec_appro.t()
        else:
            eig_value, eig_vec = torch.symeig(mean_score_mat, eigenvectors=True)

            # Selection top large eigenvectors
            r_th = threshold
            r_candidate = eig_vec.t()
            r_candidate = r_candidate.flip(0)
            eig_value_cand = eig_value.flip(0)
            max_eig_value = eig_value_cand[0]
            if num_selection is None:
                for d in range(samples1.shape[-1]):
                    eig_cand = eig_value_cand[d]
                    if torch.abs(max_eig_value / (eig_cand + 1e-10)) < r_th:
                        r_comp = r_candidate[d, :].unsqueeze(0)  # 1x D
                        if d == 0:
                            r = r_comp
                        else:
                            r = torch.cat((r, r_comp), dim=0)  # n x D
            else:
                for d in range(num_selection):
                    r_comp = r_candidate[d, :].unsqueeze(0)  # 1x D
                    if d == 0:
                        r = r_comp
                    else:
                        r = torch.cat((r, r_comp), dim=0)  # n x D
    return r

