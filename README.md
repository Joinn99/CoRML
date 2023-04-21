# CoRML
[![RecBole](https://img.shields.io/badge/RecBole-1.1.1-orange)](https://recbole.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2304.07971-red)](https://arxiv.org/abs/2304.07971) 
[![License](https://img.shields.io/github/license/Joinn99/CoRML)](https://github.com/Joinn99/CoRML/blob/torch/LICENSE.md)

[**PyTorch version**](https://github.com/Joinn99/CoRML/tree/torch) (Default) | **CuPy version**

This is the official implementation of our *The 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2023)* paper:
> Tianjun Wei, Jianghong Ma, Tommy W.S. Chow. Collaborative Residual Metric Learning. [[arXiv](https://arxiv.org/abs/2304.07971)]

<p align="center" width="100%">
    <kbd><img width="100%" src="https://raw.githubusercontent.com/Joinn99/RepositoryResource/master/CoRML.gif"> </kbd>
</p>


## Requirements
The model implementation ensures compatibility with the Recommendation Toolbox [RecBole](https://recbole.io/) (Github: [Recbole](https://github.com/RUCAIBox/RecBole)). This version employs [CuPy](https://cupy.dev/) for matrix storage and computation, and produces the results in the paper. In addition, we also offer a pure PyTorch version of CoRML model, which can be found [here](https://github.com/Joinn99/CoRML/tree/torch).  

The requirements of the running environement:

- Python: 3.8+
- PyTorch: 1.9.0+
- RecBole: 1.1.1
- CuPy: 0.10.5+

## Dataset
Here we only put zip files of datasets in the respository due to the storage limits. To use the dataset, run
```bash
unzip -o "Data/*.zip"
```
to unzip the dataset files.

If you like to test CoRML on the custom dataset, please place the dataset files in the following path:
```bash
.
|-Data
| |-[CUSTOM_DATASET_NAME]
| | |-[CUSTOM_DATASET_NAME].user
| | |-[CUSTOM_DATASET_NAME].item
| | |-[CUSTOM_DATASET_NAME].inter

```
And create `[CUSTOM_DATASET_NAME].yaml` in `./Params` with the following content:
```yaml
dataset: [CUSTOM_DATASET_NAME]
```

For the format of each dataset file, please refer to [RecBole API](https://recbole.io/docs/user_guide/data/atomic_files.html).

## Hyperparameter
For each dataset, the optimal hyperparameters are stored in `Params/[DATASET].yaml`. To tune the hyperparamters, modify the corresponding values in the file for each dataset.

The main hyparameters of CoRML are listed as follows:
 - `lambda` ($\lambda$): Weights for $\mathbf{H}$ and $\mathbf{G}$ in preference scores
 - `dual_step_length` ($\rho$): Dual step length of ADMM
 - `l2_regularization` ($\theta$): L2-regularization for learning weight matrix $\mathbf{H}$
 - `item_degree_norm` ($t$): Item degree norm for learning weight matrix $\mathbf{H}$
 - `global_scaling` ($\epsilon$): Global scaling in approximated ranking weights (in logarithm scale)
 - `user_scaling` ($t_u$): User degree scaling in approximated ranking weights

### Running on small partitions
For datasets containing a large number of items, calculating and storing the complete item-item matrix may lead to out-of-memory problem in environments with small GPU memory. Therefore, we have added the code for graph spectral partitioning to learn the item-item weight matrix on each small partitioned item set. The code was modified based on our previous work [FPSR](https://github.com/Joinn99/FPSR).

Hyperparameter `partition_ratio` is used to control the maximum ratio of the partitioned set of items relative to the complete set, ranging from 0 to 1. When `partition_ratio` is set to 1, no partitioning will be performed.

### Sparse approximation
To maintain consistency, we perform a sparse approximation of the derived matrice $\mathbf{H}$ in CoRML by setting the entries to 0 where $\lvert \mathbf{H} \rvert \leq \eta$. The threshold $\eta$ will be adjusted so that the storage size of the sparse matrix $\mathbf{H}_{Sparse}$ is less than other types of models with embedding size 64.

Hyperparameter `sparse_approx` is used to control the sparse approximation. When `sparse_approx` is set to `False`, no sparse approximation will be performed.

## Running
The script `run.py` is used to reproduced the results presented in paper. Train and evaluate CoRML on a specific dataset, run
```bash
python run.py --dataset DATASET_NAME
```

## Citation
If you wish, please cite the following paper:

```bibtex

@InProceedings{CoRML,
  author    = {{Wei}, Tianjun and {Ma}, Jianghong and {Chow}, Tommy W.~S.},
  booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  title     = {Collaborative Residual Metric Learning},
  year      = {2023},
  address   = {New York, NY, USA},
  publisher = {Association for Computing Machinery},
  series    = {SIGIR '23},
  doi       = {10.1145/3539618.3591649},
  location  = {Taipei, Taiwan},
  numpages  = {10},
  url       = {https://doi.org/10.1145/3539618.3591649},
}

@InProceedings{FPSR,
  author    = {{Wei}, Tianjun and {Ma}, Jianghong and {Chow}, Tommy W.~S.},
  booktitle = {Proceedings of the ACM Web Conference 2023},
  title     = {Fine-tuning Partition-aware Item Similarities for Efficient and Scalable Recommendation},
  year      = {2023},
  address   = {New York, NY, USA},
  publisher = {Association for Computing Machinery},
  series    = {WWW '23},
  doi       = {10.1145/3543507.3583240},
  location  = {Austin, TX, USA},
  numpages  = {11},
  url       = {https://doi.org/10.1145/3543507.3583240},
}
```
## License
This project is licensed under the terms of the MIT license.
