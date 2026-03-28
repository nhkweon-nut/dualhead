# Copyright (c) 2022, Zikang Zhou. All rights reserved.
# Modified: Ego-only cross-attention (target queries, all agents as keys/values).
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size
from torch_geometric.utils import softmax

from datasets.utils import TemporalData
from datasets.utils import init_weights

from .embedding import MultipleInputEmbedding
from .embedding import SingleInputEmbedding


class EgoOnlyInteraction(nn.Module):
    """
    Cross-attention over scene graph: only the target (forecast) agent attends to all others.
    Keys/values use every visible agent at t = historical_steps - 1; queries use only the agent row.
    Output modes are produced only for the target agent; other nodes get zero global embedding.
    """

    def __init__(
        self,
        historical_steps: int,
        embed_dim: int,
        edge_dim: int,
        num_modes: int = 6,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        rotate: bool = True,
    ) -> None:
        super().__init__()
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_modes = num_modes

        if rotate:
            self.rel_embed = MultipleInputEmbedding(in_channels=[edge_dim, edge_dim], out_channel=embed_dim)
        else:
            self.rel_embed = SingleInputEmbedding(in_channel=edge_dim, out_channel=embed_dim)
        self.layers = nn.ModuleList(
            [
                EgoOnlyInteractionLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.multihead_proj = nn.Linear(embed_dim, num_modes * embed_dim)
        self.apply(init_weights)

    def forward(
        self,
        data: TemporalData,
        local_embed: torch.Tensor,
        agent_index: torch.Tensor,
    ) -> torch.Tensor:
        t = self.historical_steps - 1
        device = local_embed.device
        # PyG Batch: 키 이름에 'index'가 있으면 collate 시 이미 노드 오프셋이 반영됨 → ptr 가산 금지
        ai_abs = agent_index.view(-1).long().to(device)

        mask = ~data["padding_mask"][:, t]
        visible = mask.nonzero(as_tuple=False).view(-1).to(device=device, dtype=torch.long)
        node_batch = getattr(data, "batch", None)
        if node_batch is None:
            node_batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        else:
            node_batch = node_batch.to(device)
        # 각 보이는 노드 → 자기 시나리오의 에이전트(글로벌 인덱스)
        tgt = ai_abs[node_batch[visible]]
        src = visible
        ego_edge_index = torch.stack([src, tgt], dim=0)

        if visible.numel() == 0:
            # 엣지 0개: MHA는 0 증분으로 두고(아래 레이어), rel_embed는 메시지 차원만 맞춘 빈 텐서
            rel_embed = torch.empty(0, self.embed_dim, device=device, dtype=local_embed.dtype)
        else:
            rel_pos = data["positions"][ego_edge_index[0], t] - data["positions"][ego_edge_index[1], t]
            if data["rotate_mat"] is None:
                rel_embed = self.rel_embed(rel_pos)
            else:
                rel_pos = torch.bmm(rel_pos.unsqueeze(-2), data["rotate_mat"][ego_edge_index[1]]).squeeze(-2)
                rel_theta = data["rotate_angles"][ego_edge_index[0]] - data["rotate_angles"][ego_edge_index[1]]
                rel_theta_cos = torch.cos(rel_theta).unsqueeze(-1)
                rel_theta_sin = torch.sin(rel_theta).unsqueeze(-1)
                rel_embed = self.rel_embed([rel_pos, torch.cat((rel_theta_cos, rel_theta_sin), dim=-1)])

        x = local_embed
        for layer in self.layers:
            x = layer(x, ego_edge_index, rel_embed, agent_idx=ai_abs)

        n = x.size(0)
        xa = self.norm(x[ai_abs])
        xa = self.multihead_proj(xa).view(-1, self.num_modes, self.embed_dim).transpose(0, 1)
        # AMP 시 Linear 출력(xa)은 fp16일 수 있고 x는 fp32인 경우가 있어 dtype은 xa에 맞춤
        out = torch.zeros(self.num_modes, n, self.embed_dim, device=device, dtype=xa.dtype)
        out[:, ai_abs, :] = xa
        return out


class EgoOnlyInteractionLayer(MessagePassing):
    """
    Same attention message as GlobalInteractorLayer, but edges are restricted to targets = agent only.
    Residual FFN is applied only to the target agent row (others unchanged).

    Note: ``update`` matches HiVT ``GlobalInteractorLayer`` (gate uses full ``x`` [N, D]).
    Ego edges only target the agent; if you need strictly agent-only matmuls inside ``update``,
    that would require a custom propagate or masking (not done here).
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1, **kwargs) -> None:
        super().__init__(aggr="add", node_dim=0, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.lin_q_node = nn.Linear(embed_dim, embed_dim)
        self.lin_k_node = nn.Linear(embed_dim, embed_dim)
        self.lin_k_edge = nn.Linear(embed_dim, embed_dim)
        self.lin_v_node = nn.Linear(embed_dim, embed_dim)
        self.lin_v_edge = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_attr: torch.Tensor,
        agent_idx: torch.Tensor,
        size: Size = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        mha_delta = self._mha_block(h, edge_index, edge_attr, size)
        x = x + mha_delta
        xa = x[agent_idx]
        xa = xa + self._ff_block(self.norm2(xa))
        out = x.clone()
        out[agent_idx] = xa
        return out

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        index: torch.Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> torch.Tensor:
        query = self.lin_q_node(x_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key_node = self.lin_k_node(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key_edge = self.lin_k_edge(edge_attr).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value_node = self.lin_v_node(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value_edge = self.lin_v_edge(edge_attr).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * (key_node + key_edge)).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return (value_node + value_edge) * alpha.unsqueeze(-1)

    def update(self, inputs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x))
        return inputs + gate * (self.lin_self(x) - inputs)

    def _mha_block(self, x: torch.Tensor, edge_index: Adj, edge_attr: torch.Tensor, size: Size) -> torch.Tensor:
        if edge_index.size(1) == 0:
            return torch.zeros_like(x)
        out = self.out_proj(self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=size))
        return self.proj_drop(out)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
