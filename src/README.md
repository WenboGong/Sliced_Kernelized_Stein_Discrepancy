# Code Structure
* `Dataloader.py`: Pytorch dataloader, Dynamically binarizeed MNIST loader.
* `Divergence.py`: Main source file implementing the divergence computation, SVGD/S-SVGD, etc.
* `GOF_Test.py`: Contains bootstrap methods for goodness-of-fit tests.
* `Kernel.py`:  Computations of kernel matrix, gradient of kernels, trace of Hessian of kernel, etc.
* `Network.py`: Includes the definitions of MLP, vanilla VAE model, VAE with implicit encoder, and Hamiltonian annealed importance sampling (HAIS) for VAE evaluation. 
* `Util.py`: Utility methods, e.g. median heuristic for bandwidth, gradient estimator from (Li. et al.)[https://arxiv.org/pdf/1705.07107.pdf]
* `active_slice`: Implements the active slice methods for finding slice directions in (Gong et al.)[https://arxiv.org/pdf/2102.03159.pdf].
* `distribution.py`: Implements the probability distributions used in experiments. 
