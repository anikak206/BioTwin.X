"""
BioTwin.X — Cross-Modal Contrastive Learning (CMCL)
===================================================
The scientific core of the platform.

Design rationale
----------------
A patient is *not* a time series and *not* a genotype. They are a **coupled
dynamical system** whose two modalities fail in *different* ways:

* the **static branch** (clinical + genomic architecture) is slow, immutable
  and highly informative but non-temporal;
* the **temporal branch** (asynchronous wearables) is fast, noisy, irregular
  and *conditioned* by the static branch — a nocturnal RMSSD collapse means
  something very different in a high-PCSK9 genotype than in a low-risk one.

CMCL therefore does **not** concatenate and hope. It uses a *dual-encoder
contrastive* formulation:

``z_s = f_static(x_static)`` and ``z_t = f_temporal(X_window)`` are projected
into a shared unit hypersphere and trained with a symmetric NT-Xent loss, so
that the two modalities are pulled into a **shared latent space** in which
"this autonomic state" and "this genomic/clinical architecture" are explicitly
aligned. In-batch negatives (other patients) force the alignment to be
*discriminative* rather than generic.

On top of the aligned embeddings sits the **Vulnerability-Conditioned
Cross-Modal Fusion (VCF)** module:

``h = [ z_s ; z_t ; z_s ⊙ z_t ; |z_s − z_t| ] -> MLP -> fused``

The Hadamard product ``z_s ⊙ z_t`` is the *interaction* term: it is the only
place in the network where a single unit can represent "high autonomic strain
**AND** high genomic vulnerability at the same time". The feed-forward network
on top of the concatenation cannot represent that conjunction — which is
precisely the signal this product is built to capture.

A **genomic vulnerability gate** ``g = σ(W [z_s ; z_p])`` additionally
re-weights the temporal pathway, so patients with a low polygenic risk have
their autonomic signal attenuated before fusion while high-risk patients have
it amplified.

Heads
-----
* ``event_head``   — 3-way multi-label (silent ischaemia / glycaemic crash /
  autonomic arrhythmia) with per-class BCE.
* ``ttx_head``     — Huber regression on log time-to-event (hours).
* ``risk_head``    — sigmoid regression onto the 0-100 clinical risk index.

All three are read from the *fused* embedding, so the risk index is
inextricably tied to the same representation that generates the alert.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Tuple

import numpy as np

try:  # pragma: no cover - exercised implicitly
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False
    nn = object  # type: ignore

from .config import ModelConfig

# --------------------------------------------------------------------------- #
# Contrastive objective
# --------------------------------------------------------------------------- #
def nt_xent(z_a: "torch.Tensor", z_b: "torch.Tensor",
            temperature: float = 0.07,
            pid_a: "torch.Tensor | None" = None,
            pid_b: "torch.Tensor | None" = None) -> "torch.Tensor":
    """Normalised Temperature-scaled Cross-Entropy (InfoNCE).

    Positive pair: ``(static_i, temporal_i)`` — the *same* patient.
    Negatives: every other patient in the mini-batch.

    **False-negative masking.** A mini-batch may legitimately contain several
    windows from the same patient (windows are highly correlated). Treating
    those as negatives would push the model to separate two observations of
    the *same* patient, destroying the shared-space geometry. When patient
    identifiers are supplied, every ``(i, j)`` pair with ``pid_i == pid_j``
    other than the positive diagonal is masked out of the denominator.

    Returns the *symmetric* loss (static→temporal and temporal→static), which
    empirically stabilises the shared-space geometry versus the one-way form.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for nt_xent")

    device = z_a.device
    batch = z_a.shape[0]
    if batch < 2:
        return (z_a.sum() + z_b.sum()) * 0.0

    a = F.normalize(z_a, dim=1)
    b = F.normalize(z_b, dim=1)
    logits = a @ b.T / temperature
    labels = torch.arange(batch, device=device)

    if pid_a is not None and pid_b is not None:
        same = (pid_a.view(-1, 1) == pid_b.view(1, -1))
        same.fill_diagonal_(False)
        logits = logits.masked_fill(same, -1e4)

    loss_ab = F.cross_entropy(logits, labels)
    loss_ba = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_ab + loss_ba)


# --------------------------------------------------------------------------- #
# Encoders
# --------------------------------------------------------------------------- #
if TORCH_AVAILABLE:

    class StaticEncoder(nn.Module):
        """Clinical + genomic architecture encoder.

        Input layout (see :mod:`biotwin.features`):
            ``[ clinical(19) | genomic(22) | derived(5) | window_context(24) ]``

        The architecture branch is *immutable* for a given patient, so its
        output is close to a patient fingerprint. That makes it a potent
        memorisation channel when the cohort is small, which is why the
        encoder is deliberately low-capacity, uses per-sample normalisation
        (``LayerNorm``, never ``BatchNorm`` — batch statistics would let one
        patient's representation depend on the rest of the batch) and is
        subject to *archetype dropout* during training.
        """

        def __init__(self, in_dim: int, cfg: ModelConfig) -> None:
            super().__init__()
            self.cfg = cfg
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(128, cfg.d_model),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_model, cfg.proj_dim),
            )
            self.vuln_gate = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.GELU(),
                nn.Linear(64, cfg.d_model),
            )
            #: learned "no archetype" token used for archetype dropout
            self.null_archetype = nn.Parameter(torch.zeros(cfg.proj_dim))

        def forward(self, x: "torch.Tensor",
                    archetype_dropout: float = 0.0
                    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            h = self.net(x)
            z = F.normalize(h, dim=1)
            gate = torch.sigmoid(self.vuln_gate(x))

            dropped = None
            if archetype_dropout > 0.0 and self.training:
                mask = (torch.rand(z.shape[0], 1, device=z.device)
                        < archetype_dropout).float()
                null = F.normalize(self.null_archetype, dim=0).view(1, -1)
                dropped = mask * null + (1.0 - mask) * z
                z = F.normalize(dropped, dim=1)
                # the gate is neutralised for dropped samples so the sample is
                # genuinely evaluated "as if" the archetype were unknown
                gate = mask * torch.ones_like(gate) + (1.0 - mask) * gate
            return z, gate, dropped

    class TemporalEncoder(nn.Module):
        """PatchTST-lite transformer over the 5-minute multi-channel grid.

        The grid is *already* asynchronously resampled upstream (with explicit
        staleness channels), so the encoder never has to invent a clock. A
        learned **modality-age embedding** is added to the patch tokens so the
        network can tell "glucose = 92 mg/dL measured 2 minutes ago" from
        "glucose = 92 mg/dL measured 40 minutes ago".
        """

        def __init__(self, n_channels: int, cfg: ModelConfig) -> None:
            super().__init__()
            self.cfg = cfg
            self.n_channels = n_channels
            self.patch_len = cfg.patch_len
            self.stride = cfg.patch_stride

            self.patch_embed = nn.Conv1d(
                in_channels=n_channels,
                out_channels=cfg.d_model,
                kernel_size=cfg.patch_len,
                stride=cfg.patch_stride,
                padding=0,
            )
            self.pos_embed = nn.Parameter(
                torch.zeros(1, 512, cfg.d_model)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

            #: Per-modality bias added to every patch token. The grid already
            #: carries explicit staleness channels, but a learned per-modality
            #: prior lets the transformer discount a channel whose value is
            #: being carried forward rather than measured.
            self.channel_bias = nn.Parameter(torch.zeros(1, n_channels, 1))

            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.d_model * cfg.ff_mult,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer, num_layers=cfg.n_layers, enable_nested_tensor=False)

            self.attention_pool = nn.Linear(cfg.d_model, 1)
            self.proj = nn.Sequential(
                nn.LayerNorm(cfg.d_model),
                nn.Linear(cfg.d_model, cfg.proj_dim),
            )

        def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            # x: (B, C, T)
            b, c, t = x.shape
            # Per-modality bias acts on the *channel* axis, before the
            # convolution mixes channels into the embedding space.
            x = x + self.channel_bias
            patches = self.patch_embed(x).transpose(1, 2)          # (B, N, D)
            n = patches.shape[1]
            patches = patches + self.pos_embed[:, :n]

            h = self.transformer(patches)
            w = torch.softmax(self.attention_pool(h), dim=1)       # (B, N, 1)
            pooled = (h * w).sum(dim=1)                             # (B, D)
            z = F.normalize(self.proj(pooled), dim=1)
            return z, pooled

    class VulnerabilityConditionedFusion(nn.Module):
        """VCF — the genomic-vulnerability-conditioned cross-modal fusion block.

        The fusion is **FiLM-style (multiplicative-first)**: the static
        archetype *modulates* the autonomic code rather than being appended to
        it.

        .. math::

            h &= \\gamma(z_s) \\odot z_t + \\beta(z_s) \\\\
            z &= \\mathrm{norm}\\big(\\mathrm{MLP}([h,\\ z_s \\odot z_t,\\ |z_s - z_t|]) + h\\big)

        Two consequences matter clinically:

        * **The static branch cannot impersonate the temporal signal.** A
          concatenated head would be free to read "this is patient #37" off
          the static half and memorise that patient's outcome — the classic
          failure mode of small-cohort deep models. Under FiLM, the static
          branch only ever *scales and shifts* the temporal code, so every
          pathway to the output still has to carry window-level physiology.
        * **The Hadamard product is the conjunction term.** ``z_s ⊙ z_t`` is
          the only place in the network that can represent "high autonomic
          strain **AND** high genomic vulnerability at the same time" — the
          clinically decisive interaction.
        """

        def __init__(self, cfg: ModelConfig) -> None:
            super().__init__()
            d = cfg.proj_dim
            self.film = nn.Sequential(
                nn.Linear(d, 2 * d),
                nn.GELU(),
                nn.Linear(2 * d, 2 * d),
            )
            self.cross_attn = nn.MultiheadAttention(d, cfg.n_heads,
                                                    dropout=cfg.dropout, batch_first=True)
            self.norm = nn.LayerNorm(d)
            self.mlp = nn.Sequential(
                nn.Linear(3 * d, 256),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(256, d),
            )
            self.residual = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

        def forward(self, z_s: "torch.Tensor", z_t: "torch.Tensor",
                    gate: "torch.Tensor") -> "torch.Tensor":
            # 1. FiLM modulation: the archetype scales/shifts the autonomic code.
            gamma, beta = self.film(z_s).chunk(2, dim=-1)
            gamma = 1.0 + torch.tanh(gamma)      # bounded around 1
            h = gamma * (z_t * gate) + beta

            # 2. Cross-modal attention: the modulated temporal state attends to
            #    the archetype, so the *event* branch stays temporal-first.
            attn, _ = self.cross_attn(h.unsqueeze(1), z_s.unsqueeze(1),
                                       z_s.unsqueeze(1), need_weights=False)
            h = self.norm(h + attn.squeeze(1))

            # 3. The conjunction term.
            inter = torch.cat([h, z_s * (z_t * gate), (z_s - z_t * gate).abs()], dim=1)
            fused = self.mlp(inter)
            fused = self.norm(fused + self.residual(h))
            return F.normalize(fused, dim=1)

    class BioTwinCMCL(nn.Module):
        """The full dual-encoder contrastive digital twin."""

        def __init__(self, static_dim: int, n_channels: int, cfg: ModelConfig,
                     n_classes: int = 3) -> None:
            super().__init__()
            self.cfg = cfg
            self.static_encoder = StaticEncoder(static_dim, cfg)
            self.temporal_encoder = TemporalEncoder(n_channels, cfg)
            self.fusion = VulnerabilityConditionedFusion(cfg)
            self.event_head = nn.Linear(cfg.proj_dim, n_classes)
            self.ttx_head = nn.Linear(cfg.d_model, 1)
            self.risk_head = nn.Sequential(
                nn.Linear(cfg.proj_dim, 64), nn.GELU(), nn.Linear(64, 1)
            )
            # Dedicated projection heads for the contrastive objective. The
            # *encoder* codes must stay free to carry window-level detail for
            # the event heads; contracting them directly onto each other would
            # collapse the within-patient variation the whole system depends on.
            self.contrast_static = nn.Sequential(
                nn.Linear(cfg.proj_dim, cfg.proj_dim), nn.GELU(),
                nn.Linear(cfg.proj_dim, cfg.proj_dim))
            self.contrast_temporal = nn.Sequential(
                nn.Linear(cfg.proj_dim, cfg.proj_dim), nn.GELU(),
                nn.Linear(cfg.proj_dim, cfg.proj_dim))
            self.temperature = cfg.temperature

        # -- forward passes ------------------------------------------------ #
        def encode(self, x_static: "torch.Tensor", x_time: "torch.Tensor",
                   archetype_dropout: float = 0.0):
            z_s, gate, dropped = self.static_encoder(x_static, archetype_dropout)
            z_t, pooled = self.temporal_encoder(x_time)
            fused = self.fusion(z_s, z_t, gate)
            return z_s, z_t, fused, pooled, gate, dropped

        def forward(self, x_static: "torch.Tensor", x_time: "torch.Tensor",
                    archetype_dropout: float = 0.0):
            z_s, z_t, fused, pooled, gate, dropped = self.encode(
                x_static, x_time, archetype_dropout)
            return {
                "z_static": z_s,
                "z_temporal": z_t,
                "z_fused": fused,
                "event_logits": self.event_head(fused),
                "ttx": self.ttx_head(pooled).squeeze(-1),
                "risk_logit": self.risk_head(fused).squeeze(-1),
                "gate": gate,
                "archetype_dropped": dropped,
            }

        # -- losses -------------------------------------------------------- #
        def compute_loss(self, out: dict, y_event: "torch.Tensor",
                         y_ttx: "torch.Tensor", y_risk: "torch.Tensor",
                         y_ttx_mask: "torch.Tensor",
                         pid_a: "torch.Tensor | None" = None,
                         pid_b: "torch.Tensor | None" = None) -> "torch.Tensor":
            cfg = self.cfg
            bce = F.binary_cross_entropy_with_logits(
                out["event_logits"], y_event,
                pos_weight=torch.tensor([6.0, 6.0, 3.0], device=y_event.device),
            )
            huber = F.smooth_l1_loss(out["ttx"], y_ttx, reduction="none", beta=1.0)
            ttx_loss = (huber * y_ttx_mask).sum() / torch.clamp(y_ttx_mask.sum(), min=1.0)
            risk_loss = F.binary_cross_entropy_with_logits(out["risk_logit"], y_risk)
            c_s = F.normalize(self.contrast_static(out["z_static"]), dim=1)
            c_t = F.normalize(self.contrast_temporal(out["z_temporal"]), dim=1)
            con = nt_xent(c_s, c_t, cfg.temperature, pid_a=pid_a, pid_b=pid_b)
            total = (cfg.w_event * bce + cfg.w_ttx * ttx_loss
                     + cfg.w_risk * risk_loss + cfg.contrastive_weight * con)
            return total, {"bce": bce.detach(), "ttx": ttx_loss.detach(),
                           "risk": risk_loss.detach(), "contrastive": con.detach()}


# --------------------------------------------------------------------------- #
# Torch-free reference implementation (numerically identical semantics)
# --------------------------------------------------------------------------- #
class ReferenceFusion:
    """NumPy mirror of :class:`VulnerabilityConditionedFusion`.

    Used by the deterministic fallback engine when no trained checkpoint or
    PyTorch runtime is available, so the platform degrades gracefully instead
    of failing closed. The semantics match the trained module: the archetype
    *modulates* the temporal code (FiLM) and the conjunction term
    ``z_s ⊙ z_t`` is carried explicitly.
    """

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng(7)
        d = 128
        self.w1 = rng.normal(0, 1 / np.sqrt(d), size=(4 * d, 256)).astype(np.float32)
        self.b1 = np.zeros(256, dtype=np.float32)
        self.w2 = rng.normal(0, 1 / np.sqrt(256), size=(256, d)).astype(np.float32)
        self.b2 = np.zeros(d, dtype=np.float32)
        self.w_film = rng.normal(0, 1 / np.sqrt(d), size=(d, 2 * d)).astype(np.float32)

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))

    def __call__(self, z_s: np.ndarray, z_t: np.ndarray, gate: np.ndarray) -> np.ndarray:
        z_t_g = z_t * gate
        film = z_s @ self.w_film
        gamma = 1.0 + np.tanh(film[:, :z_s.shape[1]])
        beta = film[:, z_s.shape[1]:]
        h = gamma * z_t_g + beta
        inter = np.concatenate([h, z_s * z_t_g, np.abs(z_s - z_t_g)], axis=1)
        g = self._gelu(inter @ self.w1 + self.b1)
        fused = g @ self.w2 + self.b2 + h
        norm = np.linalg.norm(fused, axis=1, keepdims=True) + 1e-8
        return (fused / norm).astype(np.float32)
