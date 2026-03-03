import torch
import torch.nn as nn
import torch.nn.functional as F
from FCF import FeatureComplementarityFusion  # Feature Complementarity Fusion (FCF) Module
from torch_geometric.nn import DenseGCNConv
from torch.nn import Linear, Dropout, Conv2d, MaxPool2d
from torch.autograd import Function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemporalFeatureExtractionModule(nn.Module):
    """
    Temporal Feature Extraction Module (Section 3.1).
    Effectively captures temporal patterns in EEG signals through convolution operations
    at the electrode channel scale.
    """

    def __init__(self, input_channels=16, time_steps=256):
        super(TemporalFeatureExtractionModule, self).__init__()

        self.input_channel = input_channels

        # 1D convolution applied along the temporal dimension, followed by BN and ELU
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1, time_steps // 2), stride=(1, 1)),
            nn.BatchNorm2d(3),
            nn.ELU()
        )

    def forward(self, x):
        # Temporal feature extraction
        x = x.unsqueeze(1)  # (batch, 1, 16, 256)
        temporal_feat = self.temporal_conv(x)  # (batch, 3, 16, 129)
        temporal_feat = temporal_feat.permute(0, 2, 1, 3)  # (batch, 16, 3, 129)

        # Flatten and concatenate the outputs of all filters (Eq. 2)
        temporal_feat = temporal_feat.reshape(temporal_feat.size(0), self.input_channel, -1)  # (batch, 16, 3*129)

        return temporal_feat


class ChebyshevGraphConv(nn.Module):
    """
    Graph convolution based on Chebyshev polynomials (Section 3.1).
    Captures multi-scale relationships among EEG electrodes (Eq. 5).
    """

    def __init__(self, k, in_channels, out_channels):
        super(ChebyshevGraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.weight = nn.Parameter(torch.Tensor(k * in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

    def chebyshev_polynomial(self, x, lap):
        """
        Calculate the Chebyshev polynomial recursively.
        :param x : input features
        :param lap: the normalized Laplacian matrix
        :return: the Chebyshev polynomial components
        """
        t = torch.ones(x.shape[0], x.shape[1], x.shape[2]).to(x.device)
        if self.k == 1:
            return t.unsqueeze(1)
        if self.k == 2:
            return torch.cat((t.unsqueeze(1), torch.matmul(lap, x).unsqueeze(1)), dim=1)
        elif self.k > 2:
            # T_0(L) = I
            tk_minus_one = x
            # T_1(L) = L
            tk = torch.matmul(lap, x)
            # t shape: (batch, 3, ele_channel, in_channel)
            t = torch.cat((t.unsqueeze(1), tk_minus_one.unsqueeze(1), tk.unsqueeze(1)), dim=1)
            for i in range(3, self.k):
                tk_minus_two, tk_minus_one = tk_minus_one, tk
                # T_k(L) = 2 * L * T_{k-1}(L) - T_{k-2}(L)
                tk = 2 * torch.matmul(lap, tk_minus_one) - tk_minus_two
                t = torch.cat((t, tk.unsqueeze(1)), dim=1)
            return t

    def forward(self, x, lap):
        """
        :param x: input feature matrix (batch_size, ele_channel, in_channel)
        :param lap: the normalized Laplacian matrix
        :return: the result of Graph convolution
        """
        # cp shape: (batch, k, ele_channel, in_channel)
        cp = self.chebyshev_polynomial(x, lap)
        cp = cp.permute(0, 2, 3, 1)  # (batch, ele_channel, in_channel, k)
        cp = cp.flatten(start_dim=2)  # (batch, ele_channel, in_channel * k)
        # Apply learnable weights
        out = torch.matmul(cp, self.weight)
        return out


class BiasedReLU(nn.Module):
    """
    Custom activation function with a learnable bias term to enhance nonlinearity.
    (Referred to in Figure 2 / Eq. 6)
    """

    def __init__(self, bias_shape):
        super(BiasedReLU, self).__init__()
        self.bias = nn.Parameter(torch.Tensor(1, 1, bias_shape))
        self.relu = nn.ReLU()
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.relu(self.bias + x)


def normalized_laplacian(w):
    """
    Calculate the normalized Laplacian of the adaptive adjacency matrix (Eq. 4).
    :param w: the adaptive adjacency matrix
    :return: lap: the normalized Laplacian matrix
    """
    d = torch.sum(w, dim=1)
    d_re = 1 / torch.sqrt(d + 1e-5)
    d_matrix = torch.diag_embed(d_re)
    lap = torch.eye(d_matrix.shape[0], device=w.device) - torch.matmul(torch.matmul(d_matrix, w), d_matrix)
    return lap


class BrainFunctionalConnectivityFeatureExtractionModule(nn.Module):
    """
    Brain Functional Connectivity Feature Extraction Module (Section 3.1 & Figure 2).
    Employs dynamic graph convolution with an adaptive adjacency matrix to extract
    brain functional connectivity features.
    """

    def __init__(self, num_electrodes=16, in_channels=256, num_classes=2, k=2, layers=None, dropout_rate=0.5):
        super(BrainFunctionalConnectivityFeatureExtractionModule, self).__init__()

        self.dropout_rate = dropout_rate
        self.layers = layers
        self.k = k
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.num_classes = num_classes

        if num_electrodes == 16:
            self.layers = [64]

        self.graphConvs = nn.ModuleList()
        self.graphConvs.append(ChebyshevGraphConv(self.k, self.in_channels, self.layers[0]))
        for i in range(len(self.layers) - 1):
            self.graphConvs.append(ChebyshevGraphConv(self.k, self.layers[i], self.layers[i + 1]))

        # Fully connected layer to project channel dimension to match temporal features
        self.fc = nn.Linear(64, 387, bias=True)

        # Learnable adaptive adjacency matrix (Eq. 3)
        self.adj = nn.Parameter(torch.Tensor(self.num_electrodes, self.num_electrodes))
        self.adj_bias = nn.Parameter(torch.Tensor(1))

        self.relu = nn.ReLU(inplace=True)
        self.b_relus = nn.ModuleList()

        for i in range(len(self.layers)):
            self.b_relus.append(BiasedReLU(self.layers[i]))

        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.adj)
        nn.init.trunc_normal_(self.adj_bias, mean=0, std=0.1)

    def forward(self, x):
        # Construct adaptive adjacency matrix and normalized Laplacian
        adj = self.relu(self.adj + self.adj_bias)
        lap = normalized_laplacian(adj)

        for i in range(len(self.layers)):
            x = self.graphConvs[i](x, lap)
            x = self.dropout(x)
            x = self.b_relus[i](x)  # Output shape: (batch, 16, 64)
            x = self.fc(x)

        return x


class SingleBandFeatureExtractor(nn.Module):
    """
    Single-Band Feature Extractor (Figure 1).
    Integrates the Temporal Feature Extraction Module, Brain Functional Connectivity
    Feature Extraction Module, and the FCF attention module for one specific frequency band.
    """

    def __init__(self):
        super(SingleBandFeatureExtractor, self).__init__()
        self.temporal_extractor = TemporalFeatureExtractionModule()
        self.connectivity_extractor = BrainFunctionalConnectivityFeatureExtractionModule()
        self.channels = 16
        self.heads = 4

        # Feature Complementarity Fusion (FCF) module
        self.feature_fusion = FeatureComplementarityFusion(self.channels, self.heads)

    def forward(self, x):
        temporal_feature = self.temporal_extractor(x)  # (batch, 16, 387)
        connectivity_feature = self.connectivity_extractor(x)  # (batch, 16, 64 -> projected to match)

        # Fuses temporal and connectivity features using two-stage cross-attention
        fused_feature = self.feature_fusion(temporal_feature, connectivity_feature)

        return fused_feature


class InterBandGraphInteraction(nn.Module):
    """
    频带间拓扑关系图卷积模块 (对应论文 Section 3.3)
    """

    def __init__(self, in_features: int, bn_features: int = 64, out_features: int = 32, num_bands: int = 3):
        super().__init__()
        self.num_bands = num_bands
        self.bnlin = nn.Linear(in_features, bn_features)
        self.gconv = DenseGCNConv(in_features, out_features)

    def forward(self, x):
        # 1. 通过线性变换计算频带间的相似度矩阵 (Eq. 14, 15)
        xa = torch.tanh(self.bnlin(x))
        # 使用 torch.bmm (Batch Matrix Multiply) 替代原有的 matmul，更规范
        adj = torch.softmax(torch.bmm(xa, xa.transpose(1, 2)), dim=2)

        # 2. 邻接矩阵稀疏化：仅保留 Top-K (K=2) 的边
        amask = torch.zeros_like(adj)
        _, topk_idx = adj.topk(2, dim=2)
        amask.scatter_(2, topk_idx, 1.0)
        adj = adj * amask

        # 3. 图卷积特征聚合 (Eq. 16)
        return F.relu(self.gconv(x, adj))


class FrequencyBandFusionModule(nn.Module):
    """
    精简版频带融合模块 (FFM) (对应论文 Section 3.3 & Figure 4)
    """

    def __init__(self, in_features_list=[1547 * 32, 385 * 64, 95 * 128], channels=[1, 32, 64, 128]):
        super().__init__()
        self.num_bands = 3

        self.conv_blocks = nn.ModuleList()
        self.gnn_blocks = nn.ModuleList()

        # 动态构建 3 个级联的 [Conv -> ReLU -> Dropout -> Pool -> GNN] 子模块
        for i in range(len(channels) - 1):
            # 时空特征提取分支
            self.conv_blocks.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=(1, 5)),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.MaxPool2d((1, 4))
            ))
            # 频带间交互分支
            self.gnn_blocks.append(InterBandGraphInteraction(
                in_features=in_features_list[i],
                bn_features=64,
                out_features=32,
                num_bands=self.num_bands
            ))

        self.final_drop = nn.Dropout(0.05)

    def forward(self, x):
        x = x.float()
        gnn_outputs = []

        # 遍历执行 3 个级联子模块
        for conv_block, gnn_block in zip(self.conv_blocks, self.gnn_blocks):
            # 局部时空特征提取
            x = conv_block(x)

            # 为 GNN 整理维度: (batch, channels, bands, features) -> (batch, bands, channels * features)
            b, c, bands, f = x.size()
            x_flat = x.permute(0, 2, 1, 3).reshape(b, bands, c * f)

            # 频带全局拓扑交互
            x_gnn = gnn_block(x_flat)
            gnn_outputs.append(x_gnn)

        # 拼接三个子模块的输出 (Eq. 17)
        out = torch.cat(gnn_outputs, dim=1)  # 形状: (batch, 9, 32)
        out = self.final_drop(out)
        out = out.reshape(x.size(0), -1)  # 形状: (batch, 288)

        return out


class MultiBandFeatureExtractor(nn.Module):
    """
    Main Feature Extractor of MMFDAHNet combining Multi-Band Extraction and Fusion.
    Takes multi-band EEG signals as input, processes them through independent
    Single-Band Feature Extractors, and fuses them via the Frequency Band Fusion Module.
    """

    def __init__(self):
        super(MultiBandFeatureExtractor, self).__init__()

        self.single_band_extractors = nn.ModuleList()
        # Instantiate SingleBandFeatureExtractor for each frequency band (delta, beta2, gamma)
        for i in range(3):
            single_band_extractor = SingleBandFeatureExtractor().to(device)
            setattr(self, f'single_band_extractor_{i}', single_band_extractor)
            self.single_band_extractors.append(single_band_extractor)

        self.ffm = FrequencyBandFusionModule().to(device)

    def forward(self, x):
        # Input shape: (batch, 16, 256, 3) -> Permute to (batch, 3, 16, 256)
        x = x.permute(0, 3, 1, 2)
        features = []

        # Process each frequency band independently
        for i in range(3):
            feature = x[:, i, :, :].unsqueeze(1)  # Feature shape: (batch, 1, 16, 256)
            x_fused = self.single_band_extractors[i](feature.squeeze(1))  # (batch, 16, 451)
            features.append(x_fused)

        # Stack and pass to Frequency Band Fusion Module (FFM)
        x_fused_stacked = torch.stack(features, dim=1).flatten(2)  # (batch, 3, 16*451)
        x = x_fused_stacked.unsqueeze(1)  # (batch, 1, 3, 16*451)

        out = self.ffm(x)  # Output shape: (batch, 288)

        return out


# 1. Gradient Reversal Function (Inherits from torch.autograd.Function)
class GradientReversalFunction(Function):
    """
    Multiplies the gradient by a negative constant during backpropagation
    to achieve adversarial learning.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# 2. Gradient Reversal Layer (GRL) encapsulation
class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer (GRL) mapping to the domain adversarial module.
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


# 3. Domain Discriminator (Domain Adversarial Module)
class DomainDiscriminator(nn.Module):
    """
    Domain Predictor / Domain Discriminator (Section 3.4 & Fig. 5).
    Promotes domain-invariant feature representations to mitigate domain shifts
    caused by individual differences.
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_size)
        )

    def forward(self, x, alpha=None):
        if alpha is not None:
            self.grl.alpha = alpha  # Dynamically set adversarial weight
        x = F.normalize(x, p=2, dim=1)
        x = self.grl(x)
        return self.net(x)


# 4. Spatial Cognitive Classifier
class SpatialCognitivePredictor(nn.Module):
    """
    Spatial Cognition Predictor (Section 3.4 & Fig. 5).
    Predicts spatial cognitive ability based on the fused, domain-invariant features.
    """

    def __init__(self):
        super(SpatialCognitivePredictor, self).__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(288, 64),
            nn.ELU(),
            nn.Dropout(0.4),  # Dropout layer for regularization
            nn.Linear(64, 2)  # Binary classification: High vs Low spatial cognitive ability
        )

    def forward(self, x):
        pred = self.classifier(x)
        return pred