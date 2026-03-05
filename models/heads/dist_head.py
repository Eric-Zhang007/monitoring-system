from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch import nn

from models.outputs import MultiHorizonDistOutput
from models.quality_encoder import QualityEncoder
from models.text_tower import TextTower


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    masked = scores.masked_fill(mask > 0, float("-inf"))
    probs = torch.softmax(masked, dim=dim)
    probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
    z = probs.sum(dim=dim, keepdim=True).clamp(min=1e-8)
    return probs / z


class MultiHorizonDistHead(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        horizons: Sequence[str],
        quantiles: Sequence[float],
        text_indices: Sequence[int],
        quality_indices: Sequence[int],
        d_text: int = 64,
        d_q: int = 32,
        symbol_dim: int = 16,
        regime_dim: int = 16,
        sparse_topk: int = 2,
        expert_names: Sequence[str] = ("trend", "mean_reversion", "liquidation_risk", "neutral"),
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.horizons = [str(h) for h in horizons]
        self.quantiles = [float(q) for q in quantiles]
        self.h_count = len(self.horizons)
        self.q_count = len(self.quantiles)
        self.symbol_dim = int(max(1, symbol_dim))
        self.regime_dim = int(max(1, regime_dim))
        self.expert_names = [str(x) for x in expert_names]
        self.k_expert = len(self.expert_names)
        if self.k_expert < 4:
            raise RuntimeError("moe_expert_count_must_be_at_least_4")
        self.sparse_topk = int(max(1, min(int(sparse_topk), self.k_expert)))

        self.query = nn.Parameter(torch.randn(self.h_count, self.hidden_dim) * 0.02)
        self.pool_norm = nn.LayerNorm(self.hidden_dim)
        self.ctx_proj = nn.Linear(self.hidden_dim + self.symbol_dim + d_text + d_q, self.hidden_dim)

        self.text_tower = TextTower(text_indices=text_indices, d_text=d_text)
        self.quality_encoder = QualityEncoder(quality_indices=quality_indices, d_q=d_q)

        router_in = self.regime_dim + self.symbol_dim + d_q
        self.router = nn.Sequential(
            nn.Linear(router_in, max(32, router_in)),
            nn.GELU(),
            nn.Linear(max(32, router_in), self.k_expert),
        )
        self.regime_classifier = nn.Sequential(
            nn.Linear(self.regime_dim, max(16, self.regime_dim)),
            nn.GELU(),
            nn.Linear(max(16, self.regime_dim), 3),
        )

        self.expert_mu = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for _ in range(self.k_expert)])
        self.expert_log_sigma = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for _ in range(self.k_expert)])
        self.expert_direction = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for _ in range(self.k_expert)])
        self.expert_q = nn.ModuleList([nn.Linear(self.hidden_dim, self.q_count) for _ in range(self.k_expert)])
        self.df_head = nn.Linear(self.hidden_dim, 1)

        self.register_buffer("quantile_taus", torch.tensor(self.quantiles, dtype=torch.float32), persistent=False)

    def _horizon_pool(self, h_seq: torch.Tensor, time_missing: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h_seq: [B, L, D], query: [H, D]
        attn_score = torch.einsum("bld,hd->bhl", h_seq, self.query)
        w = _masked_softmax(attn_score, time_missing.unsqueeze(1).expand(-1, self.h_count, -1), dim=-1)
        pooled = torch.einsum("bhl,bld->bhd", w, h_seq)
        return pooled, w

    @staticmethod
    def _normalize_hint(router_hint: torch.Tensor) -> torch.Tensor:
        hint = router_hint.clamp(min=0.0, max=1.0)
        s = hint.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return hint / s

    def _topk_gate(self, probs: torch.Tensor) -> torch.Tensor:
        if self.sparse_topk >= self.k_expert:
            return probs
        topv, topi = torch.topk(probs, k=self.sparse_topk, dim=-1)
        sparse = torch.zeros_like(probs)
        sparse.scatter_(dim=-1, index=topi, src=topv)
        z = sparse.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return sparse / z

    def forward(
        self,
        h_seq: torch.Tensor,
        x_values: torch.Tensor,
        x_mask: torch.Tensor,
        group_slices: Optional[Dict[str, Iterable[int]]] = None,
        *,
        static_ctx: Optional[torch.Tensor] = None,
        regime_features: Optional[torch.Tensor] = None,
        regime_mask: Optional[torch.Tensor] = None,
        router_hint: Optional[torch.Tensor] = None,
    ) -> MultiHorizonDistOutput:
        _ = group_slices
        bsz = h_seq.shape[0]
        time_missing = (x_mask.float().mean(dim=-1) >= 0.999).float()
        pooled_h, attn = self._horizon_pool(h_seq, time_missing=time_missing)
        pooled_h = self.pool_norm(pooled_h)

        text_pool = self.text_tower(x_values, x_mask)  # [B, d_text]
        q_vec = self.quality_encoder(x_values, x_mask)  # [B, d_q]
        if static_ctx is None:
            static_ctx = torch.zeros((bsz, self.symbol_dim), dtype=h_seq.dtype, device=h_seq.device)
        if static_ctx.shape != (bsz, self.symbol_dim):
            raise RuntimeError(f"static_ctx_shape_invalid:{tuple(static_ctx.shape)}:{(bsz, self.symbol_dim)}")
        if regime_features is None:
            regime_features = torch.zeros((bsz, self.regime_dim), dtype=h_seq.dtype, device=h_seq.device)
        if regime_features.shape != (bsz, self.regime_dim):
            raise RuntimeError(f"regime_features_shape_invalid:{tuple(regime_features.shape)}:{(bsz, self.regime_dim)}")
        if regime_mask is None:
            regime_mask = torch.zeros((bsz, self.regime_dim), dtype=h_seq.dtype, device=h_seq.device)
        if regime_mask.shape != (bsz, self.regime_dim):
            raise RuntimeError(f"regime_mask_shape_invalid:{tuple(regime_mask.shape)}:{(bsz, self.regime_dim)}")

        regime_obs = (1.0 - regime_mask.float()).clamp(0.0, 1.0)
        regime = regime_features * regime_obs
        pooled_ctx = torch.cat(
            [
                pooled_h,
                static_ctx.unsqueeze(1).expand(-1, self.h_count, -1),
                text_pool.unsqueeze(1).expand(-1, self.h_count, -1),
                q_vec.unsqueeze(1).expand(-1, self.h_count, -1),
            ],
            dim=-1,
        )
        mixed = torch.tanh(self.ctx_proj(pooled_ctx))

        router_input = torch.cat([regime, static_ctx, q_vec], dim=-1)
        router_logits = self.router(router_input)
        if router_hint is not None:
            if router_hint.shape != router_logits.shape:
                raise RuntimeError(f"router_hint_shape_invalid:{tuple(router_hint.shape)}:{tuple(router_logits.shape)}")
            hint = self._normalize_hint(router_hint)
            router_logits = router_logits + torch.log(hint.clamp(min=1e-6))
        expert_probs = torch.softmax(router_logits, dim=-1)
        expert_weights = self._topk_gate(expert_probs)

        expert_mu = []
        expert_log_sigma = []
        expert_dir = []
        expert_q = []
        for i in range(self.k_expert):
            mu_i = self.expert_mu[i](mixed).squeeze(-1).float()  # [B,H]
            log_sigma_i = self.expert_log_sigma[i](mixed).squeeze(-1).float().clamp(min=-7.0, max=2.0)  # [B,H]
            direction_i = self.expert_direction[i](mixed).squeeze(-1).float()  # [B,H]
            q_i = self.expert_q[i](mixed).float()  # [B,H,Q]
            expert_mu.append(mu_i)
            expert_log_sigma.append(log_sigma_i)
            expert_dir.append(direction_i)
            expert_q.append(q_i)

        mu_stack = torch.stack(expert_mu, dim=1)  # [B,K,H]
        log_sigma_stack = torch.stack(expert_log_sigma, dim=1)
        dir_stack = torch.stack(expert_dir, dim=1)
        q_stack = torch.stack(expert_q, dim=1)

        gate_w = expert_weights.unsqueeze(-1)  # [B,K,1]
        mu = (mu_stack * gate_w).sum(dim=1)  # [B,H]
        log_sigma = (log_sigma_stack * gate_w).sum(dim=1)  # [B,H]
        direction_logit = (dir_stack * gate_w).sum(dim=1)  # [B,H]
        q_delta = (q_stack * gate_w.unsqueeze(-1)).sum(dim=1)  # [B,H,Q]

        sigma = torch.exp(log_sigma).clamp(min=1e-6)

        liq_idx = self.expert_names.index("liquidation_risk")
        liq_weight = expert_weights[:, liq_idx].unsqueeze(-1)
        stress = torch.clamp(regime[:, 0:1] + torch.relu(regime[:, 3:4]) + torch.relu(-regime[:, 10:11]), min=0.0)
        sigma = sigma * (1.0 + 0.25 * liq_weight * stress)
        log_sigma = torch.log(sigma.clamp(min=1e-6))

        q_out = mu.unsqueeze(-1) + sigma.unsqueeze(-1) * q_delta
        q_out, _ = torch.sort(q_out, dim=-1)

        regime_probs = torch.softmax(self.regime_classifier(regime), dim=-1)
        df = 2.0 + torch.nn.functional.softplus(self.df_head(mixed).squeeze(-1))

        # Keep legacy text-gate diagnostics.
        text_missing = torch.ones((x_values.shape[0], 1), device=x_values.device, dtype=x_values.dtype)
        if self.text_tower.input_dim > 0:
            xm = x_mask[:, :, self.text_tower.text_indices].float().clamp(0.0, 1.0)
            coverage = 1.0 - xm.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
            text_missing = (coverage > 0).float()
        gate_legacy = text_missing.expand(-1, self.h_count)

        aux = {
            "gate": gate_legacy,
            "attention_weights": attn,
            "expert_probs": expert_probs,
            "regime": regime,
            "router_entropy": float((-expert_probs * torch.log(expert_probs.clamp(min=1e-8))).sum(dim=-1).mean().item()),
            "load_balance": expert_probs.mean(dim=0),
            "per_horizon_uncertainty": sigma,
        }
        return MultiHorizonDistOutput(
            mu=mu,
            log_sigma=log_sigma,
            q=q_out,
            direction_logit=direction_logit,
            expert_weights=expert_weights,
            regime_probs=regime_probs,
            df=df,
            aux=aux,
        )
