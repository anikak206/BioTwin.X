# BioTwin.X — Cross-Modal Contrastive Digital Twin for Predictive Cardiometabolic Prognosis

> **Happiest Health Digital Twin Challenge 2026 · proof-of-concept submission**
>
> A digital twin that fuses a patient's *static* genomic + clinical architecture
> with their *asynchronous* wearable telemetry, and predicts silent ischaemic
> and glycaemic events up to **12 hours before onset** — with an evidence chain
> a clinician can audit.

<p align="center">
  <img alt="status" src="https://img.shields.io/badge/status-research%20prototype-0ea5e9">
  <img alt="data" src="https://img.shields.io/badge/data-100%25%20synthetic-a78bfa">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-22c55e">
  <img alt="python" src="https://img.shields.io/badge/python-3.12%2B-3776ab">
</p>

> [!WARNING]
> **Research prototype — not a medical device.** Trained and evaluated on
> synthetic data only. Not validated on humans. Must not be used for diagnosis,
> triage or treatment. See [Limitations](#limitations) and
> [`docs/model_card.md`](docs/model_card.md).

---

## Table of contents

1. [Project title](#1-project-title)
2. [Team & institution](#2-team--institution)
3. [Problem statement](#3-problem-statement)
4. [Healthcare use case](#4-healthcare-use-case)
5. [Technical stack](#5-technical-stack)
6. [AI/ML model & framework details](#6-aiml-model--framework-details)
7. [Architecture diagram](#7-architecture-diagram)
8. [Presentation & demo video](#8-presentation--demo-video)
9. [License](#9-license)
10. [Quick start](#quick-start)
11. [Repository layout](#repository-layout)
12. [Honest status](#honest-status)
13. [Limitations](#limitations)

---

## 1. Project title

**BioTwin.X — Cross-Modal Contrastive Digital Twin for Predictive
Cardiometabolic Prognosis**

---

## 2. Team & institution

**Team name:** Team BioTwin.X  
**Institution:** Vellore Institute of Technology (VIT), Vellore  
**Team Members:** Anika Kumar (1 member)  
**Contact:** [anikak206@gmail.com](mailto:anikak206@gmail.com)

---

## 3. Problem statement

Cardiometabolic deterioration is **slow, silent and personal**. Three failures
define the status quo:

1. **Retroactive tracking.** Consumer wearables and Holter patches report what
   already happened — a nocturnal HRV dip, a glucose crash at 03:40 — after the
   patient has already been harmed. A report is not a warning.
2. **False-positive fatigue.** Fixed thresholds (e.g. "alert if HR > 100")
   ignore the patient's own baseline. A resting HR of 92 bpm is normal for one
   patient and an emergency for another. The resulting alert storm trains
   clinicians to dismiss the device — the single most common reason wearable
   alerting programmes fail.
3. **The genomic baseline is ignored.** The same overnight RMSSD drop is benign
   in one person and a prodrome in another, depending on their polygenic and
   clinical vulnerability. Existing systems discard that context entirely.

BioTwin.X reframes the problem: **the unit of prediction is the patient's
deviation from their own multi-modal baseline, weighted by their static
genomic/clinical architecture** — not a population threshold.

---

## 4. Healthcare use case

### Setting

A cardiometabolic clinic monitoring 100–240 patients with a continuous ECG
patch, a CGM, and a PPG wrist device. Each device streams at a different,
mutually non-commensurate cadence (1 s, 60 s, 300 s). The clinic needs a
07:00 triage list, not a raw-data lake.

### Scenario — the nocturnal RMSSD drop that matters

A 62-year-old man with elevated polygenic CVD risk (`rs1333049` dosage 2/2,
CVD PRS z = +2.3σ) wears an ECG patch and a CGM overnight.

| Time | Observation | Interpretation without context | What BioTwin.X sees |
|------|-------------|-------------------------------|---------------------|
| 23:00 | HR 68 bpm, RMSSD 47 ms | normal | within personal baseline |
| 02:10 | RMSSD 42 ms | normal | baseline drift, not yet actionable |
| **03:10** | **RMSSD 21 ms (−55 %), LF/HF 0.7, pNN50 2.1 %** | "HRV low" — ambiguous | **deep-sleep-aligned autonomic withdrawal**, *not* explained by sleep stage (his prior deep-sleep RMSSD was 41 ms) |
| 03:10 | Glucose 96 → 88 mg/dL over 3 h | normal | early sympathetic activation with preserved glycaemia |
| 03:10 | Ectopy load rising (pNN50 tail) | not scored by consumer devices | ectopic burden, invisible to mean HR |
| **03:12** | — | — | **CRI 84/100 · CRITICAL · predicted silent ischaemia in ~4.1 h** |
| 03:12 | — | — | **STAT action:** 12-lead ECG within 15 min; hold morning exercise; check troponin if symptoms |

The conjunction is the point. Low HRV alone is ambiguous; a low HRV *in a
genomically vulnerable patient, in deep sleep, with rising ectopy* is not.
Each signal alone has poor specificity; the cross-modal contrastive model is
built to represent exactly that conjunction (`z_s ⊙ z_t`).

### What the console does

* Ranks every patient by **Clinical Risk Index (0–100)** with a tier
  (`LOW` / `ELEVATED` / `HIGH` / `CRITICAL`).
* Shows the **12-hour event probabilities** (silent ischaemia, glycaemic crash,
  autonomic arrhythmia) and a predicted time-to-event.
* Emits the **AI Actionable Insights Engine** output: one primary action with
  an urgency window (`STAT` < 15 min · `URGENT` < 4 h · `PRIORITY` < 24 h ·
  `ROUTINE` < 7 d · `OPTIMISE`), plus the evidence chain that triggered it.
* Makes **silence explicit**: a LOW-risk patient says "no acute action
  required", so the absence of an alert is meaningful.

### What it is *not*

Not a diagnosis, not an autonomous escalation, not a replacement for the
clinician. It is a prioritisation and explanation layer.

---

## 5. Technical stack

| Layer | Technology | Status in this repository |
|-------|-----------|---------------------------|
| Clinician console | **Streamlit 1.66 + Plotly 7** (enterprise-ready React/Tailwind/Recharts front end is the planned V2) | ✅ `app.py` |
| Service layer | **FastAPI** async endpoints, Redis-compatible TTL cache with in-process fallback | ✅ `api/main.py` |
| Inference core | **PyTorch 2.14** Transformer time-series encoder (PatchTST-style), CPU-first | ✅ `src/biotwin/cmcl.py` |
| Static / tabular layer | **XGBoost 3.4** baseline + scikit-learn calibrated surrogate fallback | ✅ `scripts/train_pipeline.py`, `src/biotwin/inference.py` |
| Signal processing | NumPy / SciPy — per-epoch Welch PSD, RMSSD, pNN50, LF/HF, CGM variability | ✅ `src/biotwin/features.py` |
| Data | Pandas / PyArrow — long-format Parquet telemetry, memoised feature grids | ✅ `src/biotwin/dataio.py` |
| Generator | NumPy — circadian + ultradian physiology, recurrent events, sensor artefacts | ✅ `synthetic_generator.py` |
| Tests | pytest | ✅ `tests/` |
| Deployment | Docker/Kubernetes, CI, React front end | ⏳ planned (not claimed) |

### Runtime environment (reference)

```
Python 3.12+            numpy 2.5      pandas 3.0      scipy 1.18
scikit-learn 1.9        torch 2.14 (CPU)                xgboost 3.4
streamlit 1.66          plotly 7.1      pyarrow 26
```

No GPU is required. Training the `standard` cohort on a laptop CPU is a
multi-hour job; inference is sub-100 ms.

---

## 6. AI/ML model & framework details

### 6.1 Asynchronous ingestion

Seven modalities arrive at different cadences:

| Modality | Cadence | Channel(s) |
|----------|---------|-----------|
| ECG patch (R-R) | 1 s | `hr`, `hrv_rmssd`, `hrv_lf`, `hrv_hf`, `hrv_pnn50` |
| PPG | 60 s | `hr` cross-check |
| CGM | 300 s | `glucose` |
| SpO₂ | 60 s | `spo2` |
| Skin temperature | 300 s | `skin_temp` |
| Sleep staging | 30 s epochs | `sleep_stage` |

Everything is resampled to a **5-minute analysis grid** — but missingness is
never imputed away. Two explicit **asynchrony channels** (`glu_staleness`,
`rr_staleness`, in minutes) and a `coverage` channel tell the model *how old*
each value is. The temporal encoder therefore sees 12 channels, not 10.

### 6.2 Feature engineering

* **HRV** — rolling RMSSD, pNN50 (ectopy load), and **per-epoch Welch PSD**.
  SciPy 1.18's `welch` averages across segments, which destroys the per-epoch
  LF/HF resolution this pipeline needs, so `welch_psd` is implemented directly
  (50 % Hann overlap, constant detrend, one-sided power scaled by
  `fs · Σw²`) with a signal-quality gate that rejects epochs with > 20 %
  missing beats.
* **Sleep-stage alignment** — nocturnal RMSSD is compared against the
  patient's *own prior deep-sleep* RMSSD (`nhr_rmssd_dev_deep`), not against a
  population norm. This distinguishes pathological autonomic withdrawal from
  normal sleep architecture.
* **Glycaemic variability** — mean, minimum, time-in-range, hypoglycaemic
  minutes, and 3-hour slope.
* **Static vector (44 dims)** — demographics, vitals, lipids, HbA1c, fasting
  glucose, medication flags, family history, 16 SNP dosages (HWE-simulated),
  five PRS z-scores, derived interactions (insulin-resistance proxy,
  atherogenic index, metabolic syndrome, age×CVD-PRS, BMI×T2D-PRS), plus the
  24 window-context features.

**Windowing.** 12-hour look-back → 12-hour forward label. A window is positive
if a ground-truth event falls inside the forward horizon; `ttx_hours` is the
time to that event. Splits are **patient-level** — windows from one patient
never straddle train and validation.

### 6.3 Cross-modal contrastive fusion (the core contribution)

Two encoders:

```
z_s = StaticEncoder(x_static)      # low capacity, LayerNorm, archetype dropout
z_t = TemporalEncoder(x_time)      # PatchTST-style, 3 layers, 8 heads
```

**FiLM-first fusion.** The static archetype *modulates* the temporal code:

```
h        = γ(z_s) ⊙ (z_t · gate) + β(z_s),     γ = 1 + tanh(·)
a        = CrossAttention(query=h, key=z_s, value=z_s)
z_fused  = LayerNorm( MLP([h; z_s ⊙ z_t; |z_s − z_t|]) + h + a )
```

**Why this shape matters.** A concatenated head is free to read patient
identity off the static half and memorise that patient's outcome — the classic
failure mode of small-cohort deep models, and one we observed directly
(training AUC 1.000 by epoch 4, validation 0.28). Under FiLM the static branch
can only scale and shift the temporal representation, so every path to the
output still carries window-level physiology. The Hadamard term `z_s ⊙ z_t` is
the only place the network can represent *"high autonomic strain **and** high
genomic vulnerability at the same time"* — the clinically decisive
conjunction. `archetype_dropout = 0.15` randomly replaces the static vector
with a learned null archetype, further preventing lookup-table behaviour.

**Contrastive objective.** NT-Xent (InfoNCE) with temperature τ = 0.07:

```
L_contrast = − 1/N Σ_i log [ exp(sim(z_i^s, z_i^t)/τ) / Σ_{j≠i, pid_j≠pid_i} exp(sim(z_i^s, z_j^t)/τ) ]
```

Negatives are masked to **other patients** (same-patient windows are not
treated as negatives). Three projection heads (fused, static, temporal) keep
the task heads from collapsing into the contrastive space.

**Supervised heads and total loss.**

```
L = 1.0 · L_event(BCE, pos_weight=[6,6,3])
  + 0.8 · L_risk(BCE)
  + 0.4 · L_ttx(SmoothL1, masked to positive windows)
  + 0.35 · L_contrast
```

Positives are oversampled ×2 during training and corrected afterwards by
**out-of-fold temperature scaling** (no leakage).

### 6.4 Clinical Risk Index composition

```
CRI = 100 × ( 0.55·p_event + 0.25·p_class + 0.12·strain + 0.08·σ(PRS) )
```

| Term | Meaning | Why this weight |
|------|---------|-----------------|
| `p_event` | calibrated 12-h any-event probability | dominant, because it is the learned signal |
| `p_class` | probability of the dominant event type | rewards specificity, penalises undifferentiated alarm |
| `strain` | rule-based physiological strain (nocturnal RMSSD loss, LF/HF withdrawal, hypoglycaemic minutes, glucose slope), attenuated by coverage | keeps the score clinically legible and fails safe |
| `σ(PRS)` | polygenic prior | capped at 8 %: genotype shifts a score, never dominates measured physiology |

**Tiers:** `LOW` < 25 · `ELEVATED` 25–49 · `HIGH` 50–74 · `CRITICAL` ≥ 75.
**Urgency:** `STAT` < 15 min · `URGENT` < 4 h · `PRIORITY` < 24 h ·
`ROUTINE` < 7 d · `OPTIMISE` next review.

### 6.5 Targets vs. measured

The challenge targets are **ROC-AUC ≥ 0.94**, **F1 ≥ 0.91** and **p95
latency < 120 ms**. They are targets, not results. The pipeline writes every
measured number to `data/artifacts/training_report_<profile>.json` and never
fabricates a metric.

| Metric | Target | Measured (`demo`, 20 patients) | Where |
|--------|--------|-------------------------------|-------|
| Any-event ROC-AUC | ≥ 0.94 | _filled from `training_report_demo.json`_ | §6.5 |
| Ischaemia ROC-AUC | ≥ 0.94 | _…_ | |
| Crash ROC-AUC | ≥ 0.94 | _…_ | |
| Arrhythmia ROC-AUC | ≥ 0.94 | _…_ | |
| F1 (event) | ≥ 0.91 | _…_ | |
| p95 latency | < 120 ms | `python inference_engine.py benchmark` | |
| ECE | < 0.05 | _…_ | |

**Read the `standard` report, not the `demo` report, for any scientific
claim.** The demo cohort exists to make the UI runnable in a minute; 20
patients cannot support an AUC estimate. `standard` (100 patients, 104 events)
is the minimum honest evaluation, and even that is small.

### 6.6 Ablations and baselines

Every run trains, with the same budget: full fusion, temporal-only,
static-only (telemetry-derived context zeroed), XGBoost over static features
(EHR-equivalent), and XGBoost over static + engineered window summaries. The
question is whether fusion beats the best tabular baseline; if it does not,
the report says so.

### 6.7 Fail-safe backends

`BioTwinEngine` prefers the trained checkpoint (`cmcl`). If none exists it
fits a **calibrated HistGradientBoosting surrogate** over the same feature
space (`surrogate`), and if even that is impossible it answers from
deterministic clinical rules (`rule-fallback`). The platform degrades in
*accuracy*, never in *availability* — a console that shows nothing is worse
than one that shows a slightly less sharp score. The active backend is always
displayed, never silently swapped.

---

## 7. Architecture diagram

> **Placeholder — replace with the final diagram.**
> Architecture PDF: [Google Drive](https://drive.google.com/file/d/18q7UmN7LZUgVTI6OLoldBkGAzmkAOX60/view?usp=sharing)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  1 · DATA INGESTION                                                      │
│  ECG patch (1 s) · PPG (60 s) · CGM (300 s) · SpO₂ · temp · sleep        │
│  Static EHR + 16-locus SNP panel                                         │
│  → long-format Parquet (ts, modality, value, quality)                    │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │  async streams, non-commensurate cadence
┌───────────────────────────────▼──────────────────────────────────────────┐
│  2 · PROCESSING PIPELINE                                                 │
│  5-min resampling grid → 12 channels (10 physiological + 2 staleness)    │
│  per-epoch Welch PSD · RMSSD · pNN50 · LF/HF · CGM variability           │
│  sleep-stage-aligned personal baselines → 24 context features            │
│  12 h window → 12 h forward label (patient-level splits)                 │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────────┐
│  3 · AI INFERENCE CORE                                                   │
│  StaticEncoder ──► z_s ──┐                                               │
│                          ├──► FiLM fusion → cross-attention → heads      │
│  TemporalEncoder ► z_t ──┘        │                                      │
│                                   ├─► 3 event probabilities (calibrated)  │
│                                   ├─► time-to-event (hours)               │
│                                   └─► Clinical Risk Index 0–100           │
│  temperature scaling · analytic attributions · rule-based action engine  │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │  FastAPI / cached JSON
┌───────────────────────────────▼──────────────────────────────────────────┐
│  4 · DOCTOR ALERT DASHBOARD UI                                           │
│  Triage board (CRI-ranked) · patient drill-down · multi-metric charts    │
│  AI Actionable Insights Engine (STAT/URGENT/PRIORITY + evidence chain)   │
│  attribution panel · data provenance · explicit silence for LOW risk     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Presentation & demo video

> **Placeholder — replace with the final assets.**

| Asset | Link | Status |
|-------|------|--------|
| Pitch deck (PDF) | _add link_ | ⏳ |
| Demo video (15–20 min, unlisted YouTube) | _add link_ | ⏳ |
| Architecture diagram PDF | [Google Drive](https://drive.google.com/file/d/18q7UmN7LZUgVTI6OLoldBkGAzmkAOX60/view?usp=sharing) | ✅ |
| Working deck (editable) | [Google Slides](https://docs.google.com/presentation/d/1T3v8koLR-ziBnNTx24Bum1o2TTN2m6bquLUY7MnqflQ/edit?usp=sharing) | ✅ |

Suggested video outline (15–20 min): problem & clinical scenario (3 min) →
live console walkthrough on the demo cohort (5 min) → architecture & FiLM
fusion (4 min) → ablations and honest metrics (3 min) → limitations &
roadmap (2 min).

---

## 9. License

**MIT License** — Copyright (c) 2026 Team BioTwin.X (Anika Kumar and
contributors).

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

**THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.**

The full text, including the research-prototype notice, is in
[`LICENSE`](LICENSE).

---

## Quick start

### 1 · Install

```bash
git clone https://github.com/anikak206/BioTwin.X.git
cd BioTwin.X
python -m venv .venv
# Windows: .venv\Scripts\activate    |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2 · Generate the demo cohort (once)

```bash
python synthetic_generator.py --profile demo
python inference_engine.py bootstrap --profile demo
```

This creates `data/` (git-ignored): 20 patients × 7 days, ~13 M telemetry
rows, memoised feature grids and window tensors. The `standard` profile
(100 × 14 days) is the honest evaluation cohort.

### 3 · Train (optional — the console falls back to a surrogate)

```bash
python scripts/train_pipeline.py --profile demo --folds 3 --epochs 12
# headline evaluation:
python scripts/train_pipeline.py --profile standard --folds 5 --epochs 24
```

### 4 · Run the clinician console

```bash
streamlit run app.py
```

### 5 · Drive it from the CLI

```bash
python inference_engine.py cohort     --profile demo --top 15
python inference_engine.py report     --profile demo --patient BTX-0007 --context
python inference_engine.py series     --profile demo --patient BTX-0007 --hours 48
python inference_engine.py benchmark  --profile demo --samples 80
python inference_engine.py doctor     --profile demo --top 3
```

### 6 · Serve the API (optional)

```bash
pip install fastapi uvicorn
uvicorn api.main:app --port 8000
curl http://localhost:8000/v1/patients?limit=5
```

### 7 · Tests

```bash
python -m pytest              # fast unit tests
BIOTWIN_SLOW=1 python -m pytest   # + cohort-level integration tests
```

---

## Repository layout

```
BioTwin.X/
├── app.py                     # Streamlit clinician console
├── inference_engine.py        # reference CLI (report / cohort / series / benchmark)
├── synthetic_generator.py     # physiological cohort + telemetry generator
├── api/
│   └── main.py                # FastAPI async service + Redis-shaped cache
├── scripts/
│   └── train_pipeline.py      # patient-level CV, ablations, baselines, calibration
├── src/biotwin/
│   ├── config.py              # paths, profiles, hyper-parameters, SLOs
│   ├── genomics.py            # 16-locus panel, HWE genotypes, PRS, vulnerability vector
│   ├── features.py            # 5-min grid, Welch PSD, HRV/CGM features, windows
│   ├── dataio.py              # cohort bundle, caches, tensors, patient splits
│   ├── cmcl.py                # dual encoders, FiLM fusion, NT-Xent, loss
│   ├── calibration.py         # temperature scaling, ECE, Brier
│   ├── explain.py             # analytic attributions + narrative
│   ├── risk.py                # CRI tiers, action & insight rule engine
│   └── inference.py           # BioTwinEngine façade (cmcl → surrogate → rules)
├── tests/                     # genomics, features, calibration, risk, integration
├── docs/model_card.md         # intended use, data, limitations, ethics
├── data/                      # generated — git-ignored
│   ├── cohorts/  raw/  features/  models/  artifacts/
└── requirements.txt · pyproject.toml · LICENSE · README.md
```

---

## Honest status

| Component | State |
|-----------|-------|
| Synthetic generator (recurrent + imminent events, multi-event signatures) | ✅ working |
| Feature pipeline (12 channels, 24 context features, per-epoch Welch) | ✅ working |
| CMCL model (FiLM fusion, contrastive heads, calibration) | ✅ working |
| Patient-level CV, ablations, XGBoost baselines | ✅ working |
| Clinical console (triage, charts, insights, explainability) | ✅ working |
| CLI + JSON export | ✅ working |
| FastAPI service with TTL cache | ✅ working |
| Test suite | ✅ unit + opt-in integration |
| Demo checkpoint | ✅ shipped after retraining |
| `standard`-cohort 5-fold evaluation | ⏳ running / see artifacts |
| React front end, Docker, CI | ❌ not implemented (documented as future work) |
| Real-patient validation | ❌ none — synthetic only |

---

## Limitations

* **Synthetic data only.** The generator encodes our hypotheses about
  prodromal physiology; a model that learns them is learning *our
  assumptions*, not human biology. Metrics are optimistic.
* **No clinical validation.** Not a medical device. No regulatory clearance.
* **Small cohorts.** 20 patients cannot support an AUC claim; 100 is the
  minimum reported. Confidence intervals are wide.
* **Simplified genomics.** 16 loci with simulated effect weights — not a
  clinical polygenic risk score.
* **Explanations are approximations.** Analytic attributions over engineered
  features are useful for communication, not proof of causation.
* **Fairness and privacy were not evaluated.** Both are prerequisites for any
  real deployment.
* **Latency** is measured end-to-end on CPU; a production deployment needs
  batching, model quantisation and a real streaming backbone.

### Roadmap

1. External validation on PhysioNet / MIMIC-IV / OhioT1DM.
2. Real streaming ingestion (Kafka + Redis) and a containerised deploy.
3. React + Tailwind clinical front end; role-based access and audit logging.
4. Conformal prediction intervals and drift detection.
5. Fairness audit and DPIA; genetic-counselling pathway.

---

<p align="center">
  <sub>Built for the Happiest Health Digital Twin Challenge 2026 · MIT ·
  synthetic data only · not a medical device</sub>
</p>
