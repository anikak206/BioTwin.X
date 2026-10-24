"""Scratch diagnostic — not part of the shipped package."""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from biotwin.config import ModelConfig
from biotwin.dataio import CohortBundle, build_tensors
from biotwin.cmcl import BioTwinCMCL, nt_xent
from scripts.train_pipeline import Scaler, make_batches, macro_auc, predict

bundle = CohortBundle.load("demo")
tensors = build_tensors(bundle)
ids = bundle.patient_ids()
pid_index = {p: i for i, p in enumerate(ids)}
rng = np.random.default_rng(0)
order = rng.permutation(len(ids))
half = len(ids) // 2
val_ids = [ids[i] for i in order[:half]]
tr_ids = [ids[i] for i in order[half:]]
tr_idx = np.where(np.isin(tensors.patient_index, [pid_index[p] for p in tr_ids]))[0]
va_idx = np.where(np.isin(tensors.patient_index, [pid_index[p] for p in val_ids]))[0]
print(f"train windows {len(tr_idx)}  val windows {len(va_idx)}")

# ---- sklearn reference on the same static features -------------------
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
Xs = tensors.X_static
for name, col in (("ischemia", 0), ("crash", 1), ("arrhythmia", 2)):
    y = tensors.y_event[:, col]
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                       random_state=0).fit(Xs[tr_idx], y[tr_idx])
    p = m.predict_proba(Xs[va_idx])[:, 1]
    print(f"  GBM {name:12s} val AUC {roc_auc_score(y[va_idx], p):.4f}")

# ---- neural training with diagnostics --------------------------------
cfg = ModelConfig(lr=1e-3, batch_size=32)
torch.manual_seed(0)
sc = Scaler().fit(tensors.X_time[tr_idx], tensors.X_static[tr_idx])
Xt = torch.from_numpy(sc.transform_time(tensors.X_time)).float()
Xs_t = torch.from_numpy(sc.transform_static(tensors.X_static)).float()
y_e = torch.from_numpy(tensors.y_event).float()
y_r = torch.from_numpy(tensors.y_risk).float()
y_t = torch.from_numpy(tensors.y_ttx).float()
m_t = torch.from_numpy(tensors.ttx_mask).float()
pid_t = torch.from_numpy(tensors.patient_index).long()

model = BioTwinCMCL(static_dim=tensors.X_static.shape[1], n_channels=tensors.X_time.shape[1], cfg=cfg)
opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
brng = np.random.default_rng(0)
EPOCHS = 30
batches = make_batches(tr_idx, tensors.y_event, cfg.batch_size, brng)
steps = len(batches) * EPOCHS
warm = 60
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm
    else 0.5 * (1 + np.cos(np.pi * min((s - warm) / max(steps - warm, 1), 1.0))))

for ep in range(EPOCHS):
    model.train()
    t0 = time.perf_counter()
    tot = 0.0
    for bidx in make_batches(tr_idx, tensors.y_event, cfg.batch_size, brng):
        gi = bidx
        out = model(Xs_t[gi], Xt[gi])
        bce = F.binary_cross_entropy_with_logits(
            out["event_logits"], y_e[gi], pos_weight=torch.tensor([6.0, 6.0, 3.0]))
        hub = F.smooth_l1_loss(out["ttx"], y_t[gi], reduction="none", beta=1.0)
        ttx = (hub * m_t[gi]).sum() / torch.clamp(m_t[gi].sum(), min=1.0)
        rl = F.binary_cross_entropy_with_logits(out["risk_logit"], y_r[gi])
        con = nt_xent(out["z_static"], out["z_temporal"], cfg.temperature,
                      pid_a=pid_t[gi], pid_b=pid_t[gi])
        loss = cfg.w_event * bce + cfg.w_ttx * ttx + cfg.w_risk * rl + 0.35 * con
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step(); sched.step(); tot += float(loss.detach())
    # train-set AUC
    with torch.no_grad():
        model.eval()
        p_tr = torch.sigmoid(model.event_head(model(Xs_t[tr_idx], Xt[tr_idx])["z_fused"])).numpy()
        p_va = torch.sigmoid(model.event_head(model(Xs_t[va_idx], Xt[va_idx])["z_fused"])).numpy()
    tr_aucs = [roc_auc_score(tensors.y_event[tr_idx][:, k], p_tr[:, k]) for k in range(3)]
    va_aucs = [roc_auc_score(tensors.y_event[va_idx][:, k], p_va[:, k]) for k in range(3)]
    print(f"ep {ep:>3d} loss {tot/len(batches):.4f} {time.perf_counter()-t0:.1f}s "
          f"train {[f'{a:.3f}' for a in tr_aucs]} val {[f'{a:.3f}' for a in va_aucs]}")
