"""
AASIST model with swappable frontend for audio deepfake detection.

This module implements the AASIST (Audio Anti-Spoofing using Integrated
Spectro-Temporal Graph Attention Networks) backend with support for
different frontend feature extractors (wav2vec2, MiMo).

Based on:
- Hemlata Tak et al., SSL_Anti-spoofing (Speaker Odyssey 2022)
- Jee-weon Jung et al., AASIST (ICASSP 2022)
"""

from typing import Union, Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .frontends.base import BaseFrontend
from .frontends.projection import get_projection


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer for spectral/temporal feature processing."""

    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()

        # Attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        # Project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # Batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # Dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # Activation
        self.act = nn.SELU(inplace=True)

        # Temperature
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (#bs, #node, #dim)
        Returns:
            (#bs, #node, out_dim)
        """
        x = self.input_drop(x)
        att_map = self._derive_att_map(x)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x: Tensor) -> Tensor:
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map(self, x: Tensor) -> Tensor:
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_map = torch.matmul(att_map, self.att_weight)
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _project(self, x: Tensor, att_map: Tensor) -> Tensor:
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _apply_BN(self, x: Tensor) -> Tensor:
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    def _init_new_params(self, *size) -> nn.Parameter:
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    """Heterogeneous Graph Attention Layer for spectro-temporal fusion."""

    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        # Attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        # Project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        # Batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # Dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # Activation
        self.act = nn.SELU(inplace=True)

        # Temperature
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x1: Tensor, x2: Tensor, master: Optional[Tensor] = None):
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)

        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)

        x = self.input_drop(x)
        att_map = self._derive_att_map(x, num_type1, num_type2)
        master = self._update_master(x, master)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)
        return x1, x2, master

    def _update_master(self, x: Tensor, master: Tensor) -> Tensor:
        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)
        return master

    def _pairwise_mul_nodes(self, x: Tensor) -> Tensor:
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map_master(self, x: Tensor, master: Tensor) -> Tensor:
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))
        att_map = torch.matmul(att_map, self.att_weightM)
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _derive_att_map(self, x: Tensor, num_type1: int, num_type2: int) -> Tensor:
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)

        att_map = att_board
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _project(self, x: Tensor, att_map: Tensor) -> Tensor:
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _project_master(self, x: Tensor, master: Tensor, att_map: Tensor) -> Tensor:
        x1 = self.proj_with_attM(torch.matmul(att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)
        return x1 + x2

    def _apply_BN(self, x: Tensor) -> Tensor:
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    def _init_new_params(self, *size) -> nn.Parameter:
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    """Graph pooling layer with learnable top-k selection."""

    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h: Tensor) -> Tensor:
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)
        return new_h

    def top_k_graph(self, scores: Tensor, h: Tensor, k: float) -> Tensor:
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)
        h = h * scores
        h = torch.gather(h, 1, idx)
        return h


class Residual_block(nn.Module):
    """Residual block for RawNet2-style encoder."""

    def __init__(self, nb_filts: List[int], first: bool = False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(
            in_channels=nb_filts[0],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(1, 1),
            stride=1
        )
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(
            in_channels=nb_filts[1],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(0, 1),
            stride=1
        )

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(
                in_channels=nb_filts[0],
                out_channels=nb_filts[1],
                padding=(0, 1),
                kernel_size=(1, 3),
                stride=1
            )
        else:
            self.downsample = False

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        return out


class Model(nn.Module):
    """
    AASIST model with swappable frontend.

    This model combines a frontend feature extractor (wav2vec2 or MiMo)
    with the AASIST graph attention backend for audio deepfake detection.

    Args:
        frontend: Frontend feature extractor (BaseFrontend subclass)
        filts_0: Projection output dimension / AASIST input (default 128)
        encoder_scale: Scale factor for encoder channels (default 1.0)
        gat_dims: Graph attention layer dimensions
        pool_ratios: Pooling ratios for graph pooling layers
        temperatures: Temperature values for attention layers
        dropout: Final layer dropout rate
        dropout_way: Path dropout rate
        dropout_input: Input dropout rate for GAT layers (not currently used)
        projection_type: Type of projection layer ('linear' or 'mlp')
        projection_hidden_dims: Hidden dimensions for MLP projection
        projection_activation: Activation function for MLP projection
        projection_dropout: Dropout rate for MLP projection
        projection_use_batchnorm: Whether to use batch norm in MLP projection
    """

    def __init__(
        self,
        frontend: BaseFrontend,
        filts_0: int = 128,
        encoder_scale: float = 1.0,
        gat_dims: Optional[List[int]] = None,
        pool_ratios: Optional[List[float]] = None,
        temperatures: Optional[List[float]] = None,
        dropout: float = 0.5,
        dropout_way: float = 0.2,
        dropout_input: float = 0.2,
        # Projection parameters
        projection_type: str = 'linear',
        projection_hidden_dims: Optional[List[int]] = None,
        projection_activation: str = 'gelu',
        projection_dropout: float = 0.1,
        projection_use_batchnorm: bool = True,
    ):
        super().__init__()

        # Build filts from filts_0 and encoder_scale
        # Base encoder channels: [1, 32], [32, 32], [32, 64], [64, 64]
        base_channels = [[1, 32], [32, 32], [32, 64], [64, 64]]
        if encoder_scale != 1.0:
            scaled_channels = []
            for pair in base_channels:
                # Scale channels, but keep first channel of first block as 1
                if pair[0] == 1:
                    scaled_channels.append([1, int(pair[1] * encoder_scale)])
                else:
                    scaled_channels.append([int(p * encoder_scale) for p in pair])
            filts = [filts_0] + scaled_channels
        else:
            filts = [filts_0] + base_channels

        # Default AASIST parameters (can be overridden by config)
        if gat_dims is None:
            gat_dims = [64, 32]
        if pool_ratios is None:
            pool_ratios = [0.5, 0.5, 0.5, 0.5]
        if temperatures is None:
            temperatures = [2.0, 2.0, 100.0, 100.0]
        if projection_hidden_dims is None:
            projection_hidden_dims = [512, 256]

        # Store architecture params for logging/reproducibility
        self.arch_params = {
            "filts_0": filts_0,
            "encoder_scale": encoder_scale,
            "filts": filts,
            "gat_dims": list(gat_dims),
            "pool_ratios": list(pool_ratios),
            "temperatures": list(temperatures),
            "dropout": dropout,
            "dropout_way": dropout_way,
            "dropout_input": dropout_input,
            "projection_type": projection_type,
            "projection_hidden_dims": list(projection_hidden_dims),
            "projection_activation": projection_activation,
            "projection_dropout": projection_dropout,
            "projection_use_batchnorm": projection_use_batchnorm,
        }

        # Frontend (wav2vec2 or MiMo)
        self.frontend = frontend

        # Projection layer from frontend dim to AASIST input
        # Linear: simple but may lose discriminative info (original AASIST)
        # MLP: better preserves discriminative features (recommended for MiMo)
        self.LL = get_projection(
            projection_type=projection_type,
            in_dim=frontend.out_dim,
            out_dim=filts[0],
            hidden_dims=projection_hidden_dims,
            activation=projection_activation,
            dropout=projection_dropout,
            use_batchnorm=projection_use_batchnorm,
        )

        # Batch normalization and activation
        # Use encoder output channels (filts[4][1]) which scales with encoder_scale
        encoder_out_channels = filts[4][1]
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=encoder_out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)
        self.drop_way = nn.Dropout(dropout_way, inplace=True)
        self.selu = nn.SELU(inplace=True)

        # RawNet2-style encoder
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
        )

        # Attention module (scales with encoder output channels)
        attention_hidden = encoder_out_channels * 2
        self.attention = nn.Sequential(
            nn.Conv2d(encoder_out_channels, attention_hidden, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(attention_hidden),
            nn.Conv2d(attention_hidden, encoder_out_channels, kernel_size=(1, 1)),
        )

        # Position encoding
        # Frequency bins after max_pool2d(3,3) on filts_0 dimension
        freq_bins = filts[0] // 3
        self.pos_S = nn.Parameter(torch.randn(1, freq_bins, filts[-1][-1]))

        # Master nodes
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        # Graph attention layers
        self.GAT_layer_S = GraphAttentionLayer(
            filts[-1][-1], gat_dims[0], temperature=temperatures[0]
        )
        self.GAT_layer_T = GraphAttentionLayer(
            filts[-1][-1], gat_dims[0], temperature=temperatures[1]
        )

        # Heterogeneous graph attention layers
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )

        # Graph pooling layers
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        # Output layer
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Raw audio waveform (batch, samples) or (batch, samples, 1)

        Returns:
            logits: (batch, 2) - [spoof_score, bonafide_score]
        """
        # Extract features using frontend
        x_feat = self.frontend.extract_feat(x.squeeze(-1))

        # Project to AASIST input dimension
        x = self.LL(x_feat)  # (bs, frame_number, 128)

        # Post-processing on front-end features
        x = x.transpose(1, 2)  # (bs, 128, frame_number)
        x = x.unsqueeze(dim=1)  # Add channel dimension
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # RawNet2-based encoder
        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)

        # Attention
        w = self.attention(x)

        # Spectral attention
        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S[:, :m.size(2), :]

        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)

        # Temporal attention
        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)
        e_T = m1.transpose(1, 2)

        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # Learnable master nodes
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # Inference path 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1
        )
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1
        )
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # Inference path 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2
        )
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2
        )
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        # Dropout
        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        # Max fusion
        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # Readout operation
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1
        )

        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        return output

    def forward_features(self, x_feat: Tensor) -> Tensor:
        """
        Forward pass using pre-extracted features (skips frontend).

        Args:
            x_feat: Pre-extracted features (batch, time, feat_dim)

        Returns:
            logits: (batch, 2) - [spoof_score, bonafide_score]
        """
        # Project to AASIST input dimension
        x = self.LL(x_feat)  # (bs, frame_number, 128)

        # Post-processing on front-end features
        x = x.transpose(1, 2)  # (bs, 128, frame_number)
        x = x.unsqueeze(dim=1)  # Add channel dimension
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # RawNet2-based encoder
        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)

        # Attention
        w = self.attention(x)

        # Spectral attention
        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S[:, :m.size(2), :]

        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)

        # Temporal attention
        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)
        e_T = m1.transpose(1, 2)

        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # Learnable master nodes
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # Inference path 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1
        )
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1
        )
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # Inference path 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2
        )
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2
        )
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        # Dropout
        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        # Max fusion
        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # Readout operation
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1
        )

        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        return output
