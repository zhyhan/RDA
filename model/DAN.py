from typing import Optional, Sequence, Tuple, List, Dict
import torch
import torch.nn as nn
import model.backbone as backbone
import torch.nn.functional as F
# from dalib.modules.classifier import Classifier as ClassifierBase
# from dalib.modules.kernels import optimal_kernel_combinations
from numpy import array, dot
import numpy as np
from qpsolvers import solve_qp

__all__ = ['MultipleKernelMaximumMeanDiscrepancy', 'ImageClassifier']

class DANNet(nn.Module):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=256, width=256, class_num=31):
        super(DANNet, self).__init__()
        ## set base network
        self.backbone = backbone.network_dict[base_net]()
        self.use_bottleneck = use_bottleneck
        self.bottleneck_layer_list = [nn.Linear(self.backbone.output_num(), bottleneck_dim), nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck = nn.Sequential(*self.bottleneck_layer_list)
        self.head = nn.Linear(bottleneck_dim, class_num)        
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        ## initialization
        self.bottleneck[0].weight.data.normal_(0, 0.005)
        self.bottleneck[0].bias.data.fill_(0.1)
        self.head.weight.data.normal_(0, 0.01)
        self.head.bias.data.fill_(0.0)

        self.parameter_list = [{"params": self.backbone.parameters(), "lr": 0.1},
                                {"params": self.bottleneck.parameters(), "lr": 1.},
                                {"params": self.head.parameters(), "lr": 1.},]

    def forward(self, inputs):
        features = self.backbone(inputs)
        if self.use_bottleneck:
            features = self.bottleneck(features)
        logits = self.head(features)
        softmax_outputs = self.softmax(logits)

        return features, logits, softmax_outputs

class DAN(object):
    def __init__(self, base_net='ResNet50', width=1024, class_num=31, use_bottleneck=True, use_gpu=True, srcweight=3):
        self.c_net = DANNet(base_net, use_bottleneck, width, width, class_num)
        self.mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)], linear= False, quadratic_program=False)
        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = class_num
        if self.use_gpu:
            self.c_net = self.c_net.cuda()
        self.srcweight = srcweight

    def get_loss(self, inputs, labels_source):
        class_criterion = nn.CrossEntropyLoss()
        
        # define loss function
        mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
        kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
        linear= False, quadratic_program=False)

        features, logits, softmax_outputs = self.c_net(inputs)

        outputs_source = softmax_outputs.narrow(0, 0, labels_source.size(0))

        source_features = features.narrow(0, 0, labels_source.size(0))
        target_features = features.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))

        classifier_loss = class_criterion(outputs_source, labels_source)
        transfer_loss = mkmmd_loss(source_features, target_features)
        total_loss = classifier_loss + 1.0*transfer_loss
        return [total_loss, classifier_loss, transfer_loss]

    def predict(self, inputs):
        feature, _, softmax_outputs= self.c_net(inputs)
        return softmax_outputs, feature

    def get_parameter_list(self):
        return self.c_net.parameter_list

    def set_train(self, mode):
        self.c_net.train(mode)
        self.mkmmd_loss.train(mode)
        self.is_train = mode



class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    r"""The Multiple Kernel Maximum Mean Discrepancy (MK-MMD) used in
    `Learning Transferable Features with Deep Adaptation Networks <https://arxiv.org/pdf/1502.02791>`_
    Given source domain :math:`\mathcal{D}_s` of :math:`n_s` labeled points and target domain :math:`\mathcal{D}_t`
    of :math:`n_t` unlabeled points drawn i.i.d. from P and Q respectively, the deep networks will generate
    activations as :math:`\{z_i^s\}_{i=1}^{n_s}` and :math:`\{z_i^t\}_{i=1}^{n_t}`.
    The MK-MMD :math:`D_k (P, Q)` between probability distributions P and Q is defined as
    .. math::
        D_k(P, Q) \triangleq \| E_p [\phi(z^s)] - E_q [\phi(z^t)] \|^2_{\mathcal{H}_k},
    :math:`k` is a kernel function in the function space
    .. math::
        \mathcal{K} \triangleq \{ k=\sum_{u=1}^{m}\beta_{u} k_{u} \}
    where :math:`k_{u}` is a single kernel.
    Using kernel trick, MK-MMD can be computed as
    .. math::
        \hat{D}_k(P, Q) &=
        \dfrac{1}{n_s^2} \sum_{i=1}^{n_s}\sum_{j=1}^{n_s} k(z_i^{s}, z_j^{s}) \\
        &+ \dfrac{1}{n_t^2} \sum_{i=1}^{n_t}\sum_{j=1}^{n_t} k(z_i^{t}, z_j^{t}) \\
        &- \dfrac{2}{n_s n_t} \sum_{i=1}^{n_s}\sum_{j=1}^{n_t} k(z_i^{s}, z_j^{t}). \\
    Parameters:
        - **kernels** (tuple(`nn.Module`)): kernel functions.
        - **linear** (bool): whether use the linear version of DAN. Default: False
        - **quadratic_program** (bool): whether use quadratic program to solve :math:`\beta`. Default: False
    Inputs: z_s, z_t
        - **z_s** (tensor): activations from the source domain, :math:`z^s`
        - **z_t** (tensor): activations from the target domain, :math:`z^t`
    Shape:
        - Inputs: :math:`(minibatch, *)`  where * means any dimension
        - Outputs: scalar
    .. note::
        Activations :math:`z^{s}` and :math:`z^{t}` must have the same shape.
    .. note::
        The kernel values will add up when there are multiple kernels.
    Examples::
        >>> from dalib.modules.kernels import GaussianKernel
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> kernels = (GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.), GaussianKernel(alpha=2.))
        >>> loss = MultipleKernelMaximumMeanDiscrepancy(kernels)
        >>> # features from source domain and target domain
        >>> z_s, z_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> output = loss(z_s, z_t)
    """

    def __init__(self, kernels: Sequence[nn.Module], linear: Optional[bool] = False,
                 quadratic_program: Optional[bool] = False):
        super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear
        self.quadratic_program = quadratic_program

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        features = torch.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)

        if not self.quadratic_program:
            kernel_matrix = sum([kernel(features) for kernel in self.kernels])  # Add up the matrix of each kernel
            # Add 2 / (n-1) to make up for the value on the diagonal
            # to ensure loss is positive in the non-linear version
            loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)
        else:
            kernel_values = [(kernel(features) * self.index_matrix).sum() + 2. / float(batch_size - 1) for kernel in self.kernels]
            loss = optimal_kernel_combinations(kernel_values)
        return loss


def _update_index_matrix(batch_size: int, index_matrix: Optional[torch.Tensor] = None,
                         linear: Optional[bool] = True) -> torch.Tensor:
    r"""
    Update the `index_matrix` which convert `kernel_matrix` to loss.
    If `index_matrix` is a tensor with shape (2 x batch_size, 2 x batch_size), then return `index_matrix`.
    Else return a new tensor with shape (2 x batch_size, 2 x batch_size).
    """
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
    return index_matrix


class GaussianKernel(nn.Module):
    r"""Gaussian Kernel Matrix
    Gaussian Kernel k is defined by
    .. math::
        k(x_1, x_2) = \exp \left( - \dfrac{\| x_1 - x_2 \|^2}{2\sigma^2} \right)
    where :math:`x_1, x_2 \in R^d` are 1-d tensors.
    Gaussian Kernel Matrix K is defined on input group :math:`X=(x_1, x_2, ..., x_m),`
    .. math::
        K(X)_{i,j} = k(x_i, x_j)
    Also by default, during training this layer keeps running estimates of the
    mean of L2 distances, which are then used to set hyperparameter  :math:`\sigma`.
    Mathematically, the estimation is :math:`\sigma^2 = \dfrac{\alpha}{n^2}\sum_{i,j} \| x_i - x_j \|^2`.
    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and use a fixed :math:`\sigma` instead.
    Parameters:
        - sigma (float, optional): bandwidth :math:`\sigma`. Default: None
        - track_running_stats (bool, optional): If ``True``, this module tracks the running mean of :math:`\sigma^2`.
          Otherwise, it won't track such statistics and always uses fix :math:`\sigma^2`. Default: ``True``
        - alpha (float, optional): :math:`\alpha` which decides the magnitude of :math:`\sigma^2` when track_running_stats is set to ``True``
    Inputs:
        - X (tensor): input group :math:`X`
    Shape:
        - Inputs: :math:`(minibatch, F)` where F means the dimension of input features.
        - Outputs: :math:`(minibatch, minibatch)`
    """

    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))


def optimal_kernel_combinations(kernel_values: List[torch.Tensor]) -> torch.Tensor:
    # use quadratic program to get optimal kernel
    num_kernel = len(kernel_values)
    kernel_values_numpy = array([float(k.detach().cpu().data.item()) for k in kernel_values])
    if np.all(kernel_values_numpy <= 0):
        beta = solve_qp(
            P=-np.eye(num_kernel),
            q=np.zeros(num_kernel),
            A=kernel_values_numpy,
            b=np.array([-1.]),
            G=-np.eye(num_kernel),
            h=np.zeros(num_kernel),
        )
    else:
        beta = solve_qp(
            P=np.eye(num_kernel),
            q=np.zeros(num_kernel),
            A=kernel_values_numpy,
            b=np.array([1.]),
            G=-np.eye(num_kernel),
            h=np.zeros(num_kernel),
        )
    beta = beta / beta.sum(axis=0) * num_kernel  # normalize
    return sum([k * b for (k, b) in zip(kernel_values, beta)])