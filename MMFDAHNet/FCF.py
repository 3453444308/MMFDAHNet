import torch
import torch.nn as nn


class LinearAttentionMechanism(nn.Module):
    """
    Factorized Linear Attention Mechanism.
    Replaces traditional einsum implementations with explicit matrix multiplications.
    Computes attention in linear time complexity.
    """

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Explicit independent linear projections
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.out_linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_seq: torch.Tensor, k_seq: torch.Tensor, v_seq: torch.Tensor,
                extract_common: bool) -> torch.Tensor:
        """
        Inputs expect shape: [Batch, Sequence_Length, Channels]
        """
        B, L_q, _ = q_seq.shape
        L_k = k_seq.shape[1]
        L_v = v_seq.shape[1]

        # 1. Project Q, K, V
        # Shape transforms to: [B, Length, Heads, Head_Dim] -> [B, Heads, Length, Head_Dim]
        Q = self.proj_q(q_seq).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.proj_k(k_seq).view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.proj_v(v_seq).view(B, L_v, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Compute spatial softmax over Key (Normalization over sequence length)
        K_probs = torch.softmax(K, dim=2)

        # 3. Factorized Attention via explicit MatMul (Avoids einsum)
        # Context Matrix: K_probs^T @ V -> [B, Heads, Head_Dim, Head_Dim]
        K_probs_T = K_probs.transpose(-1, -2)
        context_matrix = torch.matmul(K_probs_T, V)

        # Attended Features: Q @ Context_Matrix -> [B, Heads, L_q, Head_Dim]
        attended_features = torch.matmul(Q, context_matrix)

        # 4. Routing based on the desired target representation
        if extract_common:
            # Common Information Injection
            target_features = attended_features
        else:
            # Complementary Information Extraction (Residual Difference)
            target_features = V - attended_features

        # 5. Recombine heads and project
        # [B, Heads, L_q, Head_Dim] -> [B, L_q, Heads, Head_Dim] -> [B, L_q, Channels]
        target_features = target_features.transpose(1, 2).contiguous().view(B, L_q, -1)

        output = self.out_linear(target_features)
        output = self.dropout(output)

        return output


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network using Linear layers instead of Conv1d.
    Operates on [Batch, Sequence, Channels] data format.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 3.0, dropout: float = 0.0):
        super().__init__()
        self.attention = LinearAttentionMechanism(dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.ffn = PositionwiseFeedForward(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor, extract_common: bool) -> torch.Tensor:
        """
        Inputs and Outputs strictly maintain the [B, C, L] shape used in the backbone.
        """
        # --- Format Conversion: [B, C, L] -> [B, L, C] ---
        q_seq = q_in.transpose(1, 2)
        k_seq = k_in.transpose(1, 2)
        v_seq = v_in.transpose(1, 2)

        # --- Phase 1: Attention ---
        attn_out = self.attention(q_seq, k_seq, v_seq, extract_common=extract_common)

        # Residual Connection 1
        x = q_seq + attn_out

        # --- Phase 2: LayerNorm & FFN ---
        x_norm = self.norm(x)
        ffn_out = self.ffn(x_norm)

        # Residual Connection 2
        out_seq = x + ffn_out

        # --- Format Conversion: [B, L, C] -> [B, C, L] ---
        return out_seq.transpose(1, 2)


class ChannelWiseLayerNorm(nn.Module):
    """Helper module to apply LayerNorm to the Channel dimension of [B, C, L] tensors."""

    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class FeatureComplementarityFusion(nn.Module):
    """
    Feature Complementarity Fusion (FCF) Module.

    A two-stage fusion architecture to systematically integrate temporal dynamics
    and brain functional connectivity via explicitly tailored cross-attention mechanisms.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 3.0):
        super().__init__()

        # Stage 1: Extracts complementary information (Z_diff)
        self.stage1_diff = CrossAttentionBlock(dim, num_heads, mlp_ratio)
        self.norm1 = ChannelWiseLayerNorm(dim)

        # Stage 2: Injects common spatial representation (Z_com)
        self.stage2_com = CrossAttentionBlock(dim, num_heads, mlp_ratio)
        self.norm2 = ChannelWiseLayerNorm(dim)

    def forward(self, temporal_token: torch.Tensor, brain_token: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temporal_token (torch.Tensor): Temporal features (X_t) [B, C, L_t]
            brain_token (torch.Tensor): Brain functional connectivity features (X_b) [B, C, L_b]

        Returns:
            torch.Tensor: Fused representation (X_fused) [B, C, L_b]
        """
        # =======================================================================
        # Step 1: Complementary Extraction
        # Target: Isolate unique temporal signals absent in the spatial topology.
        # Queries: Spatial features | Keys/Values: Temporal features
        # =======================================================================
        z_diff = self.stage1_diff(
            q_in=brain_token,
            k_in=temporal_token,
            v_in=temporal_token,
            extract_common=False
        )
        z_diff = self.norm1(z_diff)

        # =======================================================================
        # Step 2: Common Information Injection
        # Target: Reintroduce the core functional brain connectivity pathways.
        # Queries: Differential features | Keys/Values: Original spatial features
        # =======================================================================
        z_com = self.stage2_com(
            q_in=z_diff,
            k_in=brain_token,
            v_in=brain_token,
            extract_common=True
        )
        z_com = self.norm2(z_com)

        # =======================================================================
        # Step 3: Discriminative Integration
        # Synergistically combine the distinct and shared cognitive components.
        # =======================================================================
        x_fused = z_com + z_diff

        return x_fused