# Code Structure
* `Amortized_SVGD_MNIST.py`: Scripts for comparing VAE models trained with amortized inference (ELBO), amortized SVGD and S-SVGD (amortized maxSVGD).
* `Benchmark_GOF.py`: Scripts for toy goodness-of-fit tests with KSD, MMD and SKSD.
* `ICA_Training.py`: Scripts for training ICA model with KSD, SKSD and LSD from [Grathwohl., et al.](https://arxiv.org/pdf/2002.05616.pdf)
* `MNIST_HAIS_Eval`: Hamiltonian annealed importance sampling evaluation for trained VAE model. 
* `RBM_GOF.py`: Scripts for restricted Boltzmann machine goodness-of-fit tests with KSD, SKSD and MMD. 
# Scripts Details
**NOTE: Before running scripts, please modifies the storage path and result file names in each of the scripts.**
## Amortized SVGD MNIST
### Important Arguments
* `--method`: Training method for VAE model, including `"ELBO"`(Vanilla VAE), `"Amortized_SVGD"` (implicit encoder with SVGD) and `"Amortized_maxSVGD"` (implicit encoder with S-SVGD)
* `--latent_dim`: The latent dimensions for encoder, e.g. `32`
* `--disable_gpu`: Set this to disable the usage of gpu, e.g. `python SCRIPT_FILE.py --disable_gpu`

Other arguments can be changed in the scripts. 
### Running the script
`python Amortized_SVGD_MNIST.py --method "Amortized_SVGD" --latent_dim 32` (Use amortized SVGD to train VAE with 32 latent dimensions and GPU acceleration)

## Benchmark goodness-of-fit tests
### Important Arguments
* `--n_trials`: Number of GOF trials, default: `100`.
* `--n_samples`: Number of used samples for test, default: `1000`
* `--n_tr`: Number of training samples used for slices, default: `200`. Note that the samples used for test in SKSD is now `n_samples-n_tr` for fair comparison, e.g. `1000-200=800`.
* `--grad_opt_ep`: Number of gradient optimization (GO) epochs for finding slices. Note: This will **significantly** affect the performance, so use large epochs e.g. `1500` if active slices are not used.
* `--num_select`: Pruning. The number of slice directions used for computing SKSD (ranging from `1` to problem dimensions). Default is `-1`, meaning no pruning is used. 
* `--method`: GOF test method including `"KSD"`, `"SKSD"` and `"MMD"`
* `--distribution`: The distribution that is tested against isotropic Gaussian, including `"Laplace"`, `"multi-t"`, `"diffusion"` and `"Null"`
* `--method_score_q`: The score estimator for active slice method, including `"kernel_smooth"`, `"gradient_estimator"` and `"exact"`.
* `--method_r_opt`: Whether fixed slice direction r without update or update r. Options: `"Fixed"` or `"Not"`.
* `--optimal_init`: Set this to enable active slice method. E.g. `python SCRIPT.py --optimal_init`
* `--optimal_grad`: Set this to enable gradient-based optimization (GO) for slice directions. E.g. to purely use gradient optimization `python SCRIPT.py --optimal_init`, to use active slice + gradient optimization refinement `python SCRIPT.py --optimal_init --optimal_grad`
* `--initialization`: Initialization for slice directions, default: `"eye"` means slice directions are one-hot vectors. Other options: `"randn"`, Gaussian initialization.
* `--disable_gpu`: Set this to disable gpu acceleration.

Other arguments can be changed in the scripts.
### Running the script
Run **SKSD** to test **Laplace** against Gaussian with only **GO**, fixed direction **r**, **no pruning** and 1000 **epochs**:

`python Benchmark_GOF.py --method "SKSD" --distribution "Laplace" --optimal_grad --grad_opt_ep 1000 --method_r_opt "Fixed" --initialization "eye" --num_select -1`

Run **SKSD** to test **Laplace** against Gaussian with **active slice** using **kernel smooth** estimator, 50 epochs **GO** refinement, fixed direction **r**, pruning to **3** directions:

`python Benchmark_GOF.py --method "SKSD" --distribution "Laplace" --optimal_init --method_score_q "kernel_smooth" --optimal_grad --grad_opt_ep 50 --method_r_opt "Fixed" --initialization "eye" --num_select 3`

## ICA Model Training
### Important Arguments
* `--n_updates`: Number of ICA model update terations. Default: `15000`
* `--dim`: The dimensionality of the ICA model.
* `--method`: The ICA training method, including `"KSD"`, `"LSD"` and `"SKSD"`
* `--optimal_init`: Set this to enable active slice method for both slice directions.
* `--method_score_q`: Method for score estimation in active slice method, including `"kernel_smooth"`, `"gradient_estimator"` and `"exact"`
* `--flag_opt_r`: Set this to enable the gradient optimization of direction r. **Note: this will decrease the performance significantly.** 
* `--base_distribution`: The base distribution for ICA. Note: `"Laplace"` cannot be used with active slice method, the default is `"multi-t"`.
* `--optimal_interval1`: Active slice method uses this interval to update slice directions during the early stage training (<6000 iterations). 
* `--optimal_interval2`: Active slice method uses this interval to update slice directions during the late stage training (>6000 iterations).

### Running the script
Run **80** dimensional ICA training with **active slice method** using **kernel smooth**:

`python ICA_Training.py --dim 80 --method "SKSD" --optimal_init --method_score_q "kernel_smooth" --base_distribution "multi-t"`



## RBM goodness-of-fit test
### Important Arguments
For `--n_trials`, `--n_samples`, `--num_select`, `--grad_opt_ep`, `--method_score_q`, `--method_r_opt`, `--optimal_init`, `--optimal_grad`, and `--method`, refer to `Benchmark goodness-of-fit tests`.
* `--burn_in_samples`: Pseudo samples drawn during burn-in periods. These samples are used for finding slice directions.
* `--pseudo_setting`: Settings for pseudo samples. `"Setting1"`: Collecting pseudo samples first during burn-in, then train r,g using them (active slice method setting). `"Setting2"`: Train r,g during burn-in (Sliced kernelized Stein discrepancy paper setting).
* `--burnin`: Burn-in period.
* `--perturb`: Perturbation level list.

### Running the script
To use test RBM with perturbation **[0,0.01,0.012,0.014,0.016]**, **SKSD**,**active slice method for both directions**, **exact gradient**, **no further GO refinement**, **2000 pseudo samples**, and **pruning to 3 directions**:

`python RBM_GOT.py --num_select 3 --method_score_q "exact" --method_r_opt "Not" --optimal_init --method "SKSD" --burn_in_samples 2000 --pseudo_setting "Setting1" --perturb [0,0.01,0.012,0.014,0.016]`

