r"""
CoRML (CuPy Version)
################################################
Author:
    Tianjun Wei (tjwei2-c@my.cityu.edu.hk)
Reference:
    Tianjun Wei et al. "Collaborative Residual Metric Learning." in SIGIR 2023.
Created Date:
    2023/04/10
"""
import torch
import cupy as cp
import numpy as np
import scipy.sparse as sp

from cupy.sparse import diags
from cupyx.scipy.sparse.linalg import lobpcg

from recbole.utils import InputType
from recbole.utils.enum_type import ModelType
from recbole.model.abstract_recommender import GeneralRecommender


class SpectralInfo(object):
    r"""A class for producing spectral information of the interaction matrix, including node degrees,
    singular vectors, and partitions produced by spectral graph partitioning.

    Reference: https://github.com/Joinn99/FPSR/blob/cupy/FPSR/model.py
    """
    def __init__(self, inter_mat, config) -> None:
        # load parameters info
        self.eigen_dim = config["eigenvectors"]             # Number of eigenvectors used in SVD
        self.t_u = config["user_scaling"]
        self.norm_di = 2 * config["item_degree_norm"]
        self.partition_ratio = config["partition_ratio"]    # Maximum size ratio of item partition (1.0 means no partitioning)
        self.inter_mat = inter_mat

    def _degree(self, inter_mat=None, axis=0, exp=-0.5):
        r"""
        Degree of nodes
        """
        if inter_mat is None:
            inter_mat = self.inter_mat
        axis_sum = inter_mat.sum(axis=axis)
        d_i = cp.power(axis_sum.clip(min=1), exp).flatten()
        d_i[cp.isinf(d_i)] = 1.
        return d_i

    def _svd(self, mat, k):
        r"""
        Truncated singular value decomposition (SVD)
        """
        _, V = lobpcg(
            mat.T @ mat, cp.random.rand(mat.shape[1], k), largest=True)
        return V

    def _norm_adj(self, ilist=None):
        r"""
        Normalized adjacency matrix
        """
        if ilist is None:
            normed = diags(self._degree(axis=1)) @ self.inter_mat @ \
                diags(self.di_isqr.flatten())
        else:
            inter_mat = self.inter_mat[:, ilist]
            normed = diags(self._degree(inter_mat, axis=1)) @ inter_mat @ \
                diags(self.di_isqr.flatten()[ilist])
        return normed

    def run(self):
        r"""
        Spectral information
        """
        self.di_isqr = self._degree(axis=0).reshape(-1, 1)
        self.di_sqr = self._degree(axis=0, exp=0.5).reshape(1, -1)

        u_norm = self._degree(axis=1, exp=-self.t_u).reshape(-1, 1)
        self.u_norm = u_norm / u_norm.min()

        self.V_mat = self._svd(self._norm_adj(), self.eigen_dim)

        return self.di_sqr, self.u_norm, self.V_mat

    def _partitioning(self, V) -> torch.Tensor:
        r"""
        Graph biparitioning
        """
        split = cp.asnumpy(V[:, 1] >= 0)
        if split.sum() == split.shape[0] or split.sum() == 0:
            split = cp.asnumpy(V[:, 1] >= cp.median(V[:, 1]))
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
            split = self._partitioning(self._svd(self._norm_adj(ilist), 2))
            return self.get_partition(ilist[np.where(split)], total_num) + \
                self.get_partition(ilist[np.where(~split)], total_num)


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
        cp.random.seed(config["seed"])

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
        self.inter_mat = cp.sparse.csc_matrix(dataset.inter_matrix(
            form="csr"), dtype=cp.float32)   # User-item interaction matrix

        # TRAINING PROCESS
        item_list = self.update_G(config)
        self.update_H(item_list)

    def DI(self, pow=1., ilist=None):
        r"""
        Degree of item node
        """
        if ilist is not None:
            return cp.power(self.di_sqr[:, ilist], pow)
        else:
            return cp.power(self.di_sqr, pow)

    def update_G(self, config):
        r"""
        Update G matrix
        """
        G = SpectralInfo(self.inter_mat, config)
        self.di_sqr, self.u_norm, self.V_mat = G.run()

        item_list = G.get_partition(
            np.arange(self.n_items), self.n_items
        )
        return item_list

    def update_H(self, item_list):
        r"""
        Update H matrix
        """
        self.H_dict = sp.dok_matrix((self.n_items, self.n_items),
                                    dtype=np.float32)
        for ilist in item_list:
            H_triu = self.update_H_part(ilist)
            H_triu.data *= H_triu.data >= 5e-4         # Filter out small values
            H_triu.eliminate_zeros()
            self.H_dict._update(
                dict(
                    zip(
                        zip(ilist[H_triu.row], ilist[H_triu.col]),
                        H_triu.data
                    )
                )
            )
        self.H_mat = cp.sparse.csr_matrix(self.H_dict.tocsr(), dtype=cp.float32)
        # Sparse approximation
        if self.sparse_approx:
            limit = (self.n_users + self.n_items) * 64 # Embedding size in MF models
            thres = 1e-4
            while (2 * self.H_mat.nnz + self.n_items) >= limit:
                self.H_mat.data *= (self.H_mat.data > thres)
                thres *= 1.25
                self.H_mat.eliminate_zeros()

    def update_H_part(self, ilist):
        r"""
        Learning H in each partition (if any)
        """
        R_mat = self.inter_mat[:, ilist] @ diags(
            self.DI(-self.norm_di, ilist).flatten())

        H_aux = (
            (0.5 / self.lambda_) * (R_mat.T @
                                    R_mat).todense()
        ).astype(cp.float32)

        II_mat = (
            R_mat.T @ (diags(self.u_norm.flatten()) @ R_mat)
        ).todense()
        del R_mat

        V_mat = self.V_mat[ilist, :]
        diag_vvt = cp.square(V_mat).sum(axis=1).flatten()

        G_mat = (
            diags(diag_vvt) - self.DI(self.norm_di - 1, ilist).T *
            (V_mat @ V_mat.T).clip(0) * self.DI(1 - self.norm_di, ilist)
        ).astype(cp.float32)
        del V_mat, diag_vvt

        H_aux = H_aux + (
            self.eps * ((1 / self.lambda_) - 1) *
            (II_mat @ G_mat)
        ).astype(cp.float32)
        del G_mat

        II_inv = cp.linalg.inv(
            self.eps * II_mat + diags(
                self.DI(2, ilist).flatten() * self.theta + self.rho
            )
        ).astype(cp.float32)
        del II_mat

        H_aux = II_inv @ H_aux
        Phi_mat = cp.zeros_like(II_inv, dtype=cp.float32)
        S_mat = cp.zeros_like(II_inv, dtype=cp.float32)

        for _ in range(self.admm_iter):
            # Iteration
            H_tilde = H_aux + II_inv @ (self.rho * (S_mat - Phi_mat))
            lag_op = cp.diag(H_tilde) / (cp.diag(II_inv) + 1e-10)
            H_mat = H_tilde - II_inv * lag_op                   # Update H
            S_mat = H_mat + Phi_mat                             
            S_mat = ((S_mat.T + S_mat) / 2).clip(0)             # Update S
            Phi_mat += H_mat - S_mat                            # Update Phi

        return sp.triu(sp.coo_matrix(cp.asnumpy(S_mat)))

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
        user = cp.array(interaction[self.USER_ID].cpu().numpy())
        R_mat = self.inter_mat[user, :].toarray()

        Y_mat = R_mat * \
            self.DI(-1) @ self.V_mat @ self.V_mat.T * self.di_sqr
        Y_mat = ((1 / self.lambda_) - 1) * \
            (Y_mat - R_mat * cp.square(self.V_mat).sum(axis=1).reshape(1, -1)).clip(0)

        R_mat = R_mat @ diags(self.DI(-self.norm_di).flatten())
        Y_mat += (R_mat @ self.H_mat + R_mat @ self.H_mat.T) @ \
            diags(self.DI(self.norm_di).flatten())

        return torch.from_numpy(cp.asnumpy(Y_mat.flatten()))
