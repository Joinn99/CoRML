r"""
CoRML (PyTorch Version)
################################################
Author:
    Tianjun Wei (tjwei2-c@my.cityu.edu.hk)
Reference:
    Tianjun Wei et al. "Collaborative Residual Metric Learning." in SIGIR 2023.
Created Date:
    2023/04/10
"""
import torch
import numpy as np

from recbole.utils import InputType
from recbole.utils.enum_type import ModelType
from recbole.model.abstract_recommender import GeneralRecommender


class SpectralInfo(object):
    r"""A class for producing spectral information of the interaction matrix, including node degrees,
    singular vectors, and partitions produced by spectral graph partitioning.

    Reference: https://github.com/Joinn99/FPSR/blob/torch/FPSR/model.py
    """
    def __init__(self, inter_mat, config) -> None:
        # load parameters info
        self.eigen_dim = config["eigenvectors"]         # Number of eigenvectors used in SVD
        self.t_u = config["user_scaling"]
        self.norm_di = 2 * config["item_degree_norm"]
        self.partition_ratio = config["partition_ratio"]            # Maximum size ratio of item partition (1.0 means no partitioning)
        self.inter_mat = inter_mat

    def _degree(self, inter_mat=None, dim=0, exp=-0.5) -> torch.Tensor:
        r"""Get the degree of users and items.
        
        Returns:
            Tensor of the node degrees.
        """
        if inter_mat is None:
            inter_mat = self.inter_mat
        d_inv = torch.nan_to_num(
            torch.clip(torch.sparse.sum(inter_mat, dim=dim).to_dense(), min=1.).pow(exp), nan=1., posinf=1., neginf=1.
        )
        return d_inv

    def _svd(self, mat, k) -> torch.Tensor:
        r"""Perform Truncated singular value decomposition (SVD) on
        the input matrix, return top-k eigenvectors.
        
        Returns:
            Tok-k eigenvectors.
        """
        _, _, V = torch.svd_lowrank(mat, q=max(4*k, 32), niter=10)
        return V[:, :k]

    def _norm_adj(self, item_list=None) -> torch.Tensor:
        r"""Get the normalized item-item adjacency matrix for a group of items.
        
        Returns:
            Sparse tensor of the normalized item-item adjacency matrix.
        """
        if item_list is None:
            vals = self.inter_mat.values() * self.di_isqr[self.inter_mat.indices()[1]].squeeze()
            return torch.sparse_coo_tensor(
                            self.inter_mat.indices(),
                            self._degree(dim=1)[self.inter_mat.indices()[0]] * vals,
                            size=self.inter_mat.shape, dtype=torch.float
                        ).coalesce()
        else:
            inter = self.inter_mat.index_select(dim=1, index=item_list).coalesce()
            vals = inter.values() * self.di_isqr[item_list][inter.indices()[1]].squeeze()
            return torch.sparse_coo_tensor(
                            inter.indices(),
                            self._degree(inter, dim=1)[inter.indices()[0]] * vals,
                            size=inter.shape, dtype=torch.float
            ).coalesce()

    def run(self):
        r"""
        Spectral information
        """
        self.di_isqr = self._degree(dim=0).reshape(-1, 1)
        self.di_sqr = self._degree(dim=0, exp=0.5).reshape(1, -1)

        u_norm = self._degree(dim=1, exp=-self.t_u).reshape(-1, 1)
        self.u_norm = u_norm / u_norm.min()

        self.V_mat = self._svd(self._norm_adj(), self.eigen_dim)

        return self.di_sqr, self.u_norm, self.V_mat

    def partitioning(self, V) -> torch.Tensor:
        r"""
        Graph bipartitioning
        """
        split = V[:, 1] >= 0
        if split.sum() == split.shape[0] or split.sum() == 0:
            split = V[:, 1] >= torch.median(V[:, 1])
        return split

    def get_partition(self, ilist, total_num):
        r"""
        Get partitions of item-item graph
        """
        assert self.partition_ratio > (1 / total_num)

        if ilist.shape[0] <= total_num * self.partition_ratio:
            return [ilist]
        else:
            # If the partition size is larger than size limit,
            # perform graph partitioning on this partition.
            split = self.partitioning(self._svd(self._norm_adj(ilist), 2))
            return self.get_partition(ilist[torch.where(split)[0]], total_num) + \
                self.get_partition(ilist[torch.where(~split)[0]], total_num)


class CoRML(GeneralRecommender):
    r"""CoRML is an item-based metric learning model for collaborative filtering.

    CoRML learn a generalized distance user-item distance metric to capture user
    preference in user-item interaction signals by modeling the residuals of general
    Mahalanobis distance.
    """
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        r"""
        Model initialization and training.
        """
        super().__init__(config, dataset)

        self.lambda_ = config["lambda"]                     # Weights for H and G in preference scores
        self.rho = config["dual_step_length"]               # Dual step length of ADMM
        self.theta = config["l2_regularization"]            # L2-regularization for learning weight matrix H
        self.norm_di = 2 * config["item_degree_norm"]       # Item degree norm for learning weight matrix H
        self.eps = np.power(10, config["global_scaling"])   # Global scaling in approximated ranking weights (in logarithm scale)

        self.sparse_approx = config["sparse_approx"]        # Sparse approximation to reduce storage size of H
        self.admm_iter = config["admm_iter"]                # Number of iterations for ADMM

        # Dummy pytorch parameters required by Recbole
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        # User-item interaction matrix
        self.inter_mat = dataset.inter_matrix(form='coo')
        self.inter_mat = torch.sparse_coo_tensor(
                            torch.LongTensor(np.array([self.inter_mat.row, self.inter_mat.col])),
                            torch.FloatTensor(self.inter_mat.data),
                            size=self.inter_mat.shape, dtype=torch.float
                        ).coalesce().to(self.device)

        # TRAINING PROCESS
        item_list = self.update_G(config)
        self.update_H(item_list)

    def DI(self, pow=1., ilist=None):
        r"""
        Degree of item node
        """
        if ilist is not None:
            return torch.pow(self.di_sqr[:, ilist], pow)
        else:
            return torch.pow(self.di_sqr, pow)

    def update_G(self, config):
        r"""
        Update G matrix
        """
        G = SpectralInfo(self.inter_mat, config)
        self.di_sqr, self.u_norm, self.V_mat = G.run()
        item_list = G.get_partition(
            torch.arange(self.n_items, device=self.device), self.n_items
        )
        return item_list

    def update_H(self, item_list):
        r"""
        Update H matrix
        """
        self.H_indices = []
        self.H_values = []

        for ilist in item_list:
            H_triu = self.update_H_part(ilist)
            H_triu = torch.where(H_triu >= 5e-4, H_triu, 0).to_sparse_coo()
            self.H_indices.append(ilist[H_triu.indices()])
            self.H_values.append(H_triu.values())
        
        H_mat = torch.sparse_coo_tensor(indices=torch.cat(self.H_indices, dim=1),
                                         values=torch.cat(self.H_values, dim=0),
                                         size=(self.n_items, self.n_items)).coalesce()
        del self.H_indices, self.H_values
        
        # Sparse approximation
        if self.sparse_approx:
            limit = (self.n_users + self.n_items) * 64 # Embedding size in MF models
            thres = 1e-4
            while (H_mat._nnz() + H_mat.indices().shape[-1] + self.n_items + 1) >= limit:
                mask = torch.where(H_mat.values() > thres)[0]
                H_mat = torch.sparse_coo_tensor(indices=H_mat.indices().index_select(-1, mask),
                                            values=H_mat.values().index_select(-1, mask),
                                            size=(self.n_items, self.n_items)).coalesce()
                thres *= 1.25
        self.H_mat = H_mat.T.to_sparse_csr()

    def _inner_prod(self, A_mat: torch.Tensor, B_mat: torch.Tensor):
        r"""
        Small-batch inner product
        """
        assert A_mat.shape[-2] == B_mat.shape[-2]
        result = torch.zeros((A_mat.shape[-1], B_mat.shape[-1]), device=self.device)
        for chunk in torch.split(torch.arange(0, A_mat.shape[-2], device=self.device), 10000):
            result += A_mat.index_select(dim=-2, index=chunk).to_dense().T @ \
                      B_mat.index_select(dim=-2, index=chunk).to_dense()
        return result

    def update_H_part(self, ilist):
        r"""
        Learning H in each partition (if any)
        """
        R_mat = self.inter_mat.index_select(dim=1, index=ilist) * self.DI(-self.norm_di, ilist)

        H_aux = (0.5 / self.lambda_) * self._inner_prod(R_mat, R_mat)
        II_mat = self._inner_prod(R_mat, self.u_norm * R_mat)
        del R_mat

        V_mat = self.V_mat[ilist, :]
        diag_vvt = torch.square(V_mat).sum(dim=1).view(-1)

        G_mat = - self.DI(self.norm_di - 1, ilist).T * \
            (V_mat @ V_mat.T).clip(0).fill_diagonal_(0) * self.DI(1 - self.norm_di, ilist)
        del V_mat, diag_vvt

        H_aux = H_aux + (
            self.eps * ((1 / self.lambda_) - 1) *
            (II_mat @ G_mat)
        )
        del G_mat

        II_inv = torch.inverse(
            self.eps * II_mat + torch.diag(
                self.DI(2, ilist).view(-1) * self.theta + self.rho
            )
        )
        del II_mat

        H_aux = II_inv @ H_aux
        Gamma_mat = torch.zeros_like(H_aux, device=self.device)
        S_mat = torch.zeros_like(H_aux, device=self.device)

        for _ in range(self.admm_iter):
            # ADMM Iteration
            H_tilde = H_aux + II_inv @ (self.rho * (S_mat - Gamma_mat))
            lag_op = torch.diag(H_tilde) / (torch.diag(II_inv) + 1e-10)
            H_mat = H_tilde - II_inv * lag_op                   # Update H
            S_mat = H_mat + Gamma_mat                             
            S_mat = torch.clip((S_mat.T + S_mat) / 2, min=0)    # Update S
            Gamma_mat += H_mat - S_mat                          # Update Phi

        return torch.triu(S_mat)

    def forward(self):
        r"""
        Abstract method of GeneralRecommender in RecBole (not used)
        """
        pass

    def calculate_loss(self, interaction):
        r"""
        Abstract method of GeneralRecommender in RecBole (not used)
        """
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        r"""
        Abstract method of GeneralRecommender in RecBole (not used)
        """
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        r"""
        Recommend items for the input users
        """
        R_mat = self.inter_mat.index_select(dim=0, index=interaction[self.USER_ID]).to_dense()

        Y_mat = R_mat * self.DI(-1) @ self.V_mat @ self.V_mat.T * self.di_sqr
        Y_mat = ((1 / self.lambda_) - 1) * \
            torch.clip(Y_mat - R_mat * torch.square(self.V_mat).sum(dim=1).reshape(1, -1), min=0)

        R_mat = R_mat * self.DI(-self.norm_di)
        Y_mat += ((self.H_mat @ R_mat.T).T + R_mat @ self.H_mat) * self.DI(self.norm_di)

        return Y_mat
