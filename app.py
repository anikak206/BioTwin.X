"""
BioTwin.X — Clinician Console
=============================
Streamlit front-end for the cross-modal contrastive digital twin.

Design intent
-------------
This is a **triage console**, not a dashboard. A clinician opening it at 07:00
should answer three questions in under ten seconds:

1. *Who needs me first?*      → the board, ranked by Clinical Risk Index.
2. *Why this patient?*        → the evidence chain, not a bare score.
3. *What do I do next?*       → the AI Actionable Insights Engine, which emits
                                 an explicit, urgency-tagged clinical action.

Everything else — embeddings, ablations, calibration, provenance — is one
expander away, because a model that cannot be interrogated does not belong in
a clinical workflow.

Run
---
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from biotwin.config import (  # noqa: E402
    DEFAULT_PROFILE,
    MODALITIES,
    PROFILES,
    SLO_F1,
    SLO_LATENCY_MS,
    SLO_ROC_AUC,
)
from biotwin.dataio import patient_features  # noqa: E402
from biotwin.inference import BioTwinEngine, PatientReport  # noqa: E402
from biotwin.risk import TIER_COLORS  # noqa: E402

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
INK = "#0b1220"
PANEL = "#111a2e"
PANEL_2 = "#16213a"
GRID = "#243049"
TEXT = "#e6edf7"
MUTED = "#8fa1bd"
ACCENT = "#38bdf8"
VIOLET = "#a78bfa"
GREEN = "#22c55e"
AMBER = "#f59e0b"
RED = "#ef4444"

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, family="Inter, system-ui, sans-serif", size=12),
    margin=dict(l=8, r=8, t=28, b=8),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                bgcolor="rgba(0,0,0,0)"),
)

CSS = f"""
<style>
  .stApp {{ background: radial-gradient(1200px 600px at 15% -10%, #17233d 0%, {INK} 55%); }}
  section[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {GRID}; }}
  h1, h2, h3 {{ letter-spacing: -0.02em; }}
  .btx-hero {{
    display:flex; align-items:baseline; gap:14px; padding: 4px 0 18px 0;
    border-bottom: 1px solid {GRID}; margin-bottom: 18px;
  }}
  .btx-hero .mark {{ font-size: 26px; font-weight: 800; letter-spacing: -0.03em; }}
  .btx-hero .mark span {{ color: {ACCENT}; }}
  .btx-hero .sub {{ color: {MUTED}; font-size: 13px; }}
  .btx-chip {{
    display:inline-block; padding: 2px 9px; border-radius: 999px;
    font-size: 11px; font-weight: 700; letter-spacing: .04em;
  }}
  .btx-card {{
    background: linear-gradient(180deg, {PANEL_2} 0%, {PANEL} 100%);
    border: 1px solid {GRID}; border-radius: 14px; padding: 16px 18px;
    margin-bottom: 12px;
  }}
  .btx-card .urgency {{
    font-size: 11px; font-weight: 800; letter-spacing: .08em;
  }}
  .btx-card .title {{ font-size: 15px; font-weight: 700; margin: 4px 0 6px 0; }}
  .btx-card .rationale {{ color: {MUTED}; font-size: 13px; line-height: 1.5; }}
  .btx-card ul {{ margin: 8px 0 0 0; padding-left: 18px; }}
  .btx-card li {{ color: {TEXT}; font-size: 12.5px; line-height: 1.55; }}
  .btx-kpi {{
    background: {PANEL}; border: 1px solid {GRID}; border-radius: 14px;
    padding: 14px 16px;
  }}
  .btx-kpi .label {{ color: {MUTED}; font-size: 11px; letter-spacing: .08em;
                    text-transform: uppercase; }}
  .btx-kpi .value {{ font-size: 27px; font-weight: 800; line-height: 1.2; }}
  .btx-kpi .hint {{ color: {MUTED}; font-size: 11.5px; }}
  .btx-disclaimer {{
    color: {MUTED}; font-size: 11.5px; border-top: 1px solid {GRID};
    padding-top: 12px; margin-top: 26px; line-height: 1.6;
  }}
  div[data-testid="stMetricValue"] {{ font-size: 24px; }}
  .stTabs [data-baseweb="tab"] {{ font-size: 13.5px; }}
</style>
"""


def inject_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_engine(profile: str, model_mtime: float, allow_surrogate: bool = True) -> BioTwinEngine:
    """One engine per (profile, checkpoint version) for the whole server.

    ``model_mtime`` is part of the cache key on purpose: retraining a
    checkpoint must not require a server restart.
    """
    return BioTwinEngine(profile=profile, allow_surrogate=allow_surrogate, verbose=False)


@st.cache_data(show_spinner=False)
def get_board(profile: str, model_mtime: float) -> pd.DataFrame:
    return get_engine(profile, model_mtime).score_cohort()


@st.cache_data(show_spinner=False)
def get_series(profile: str, model_mtime: float, patient_id: str,
               hours: float) -> pd.DataFrame:
    return get_engine(profile, model_mtime).risk_series(patient_id, hours=hours, step_minutes=30)


@st.cache_data(show_spinner=False)
def get_grid(profile: str, patient_id: str) -> pd.DataFrame:
    return patient_features(patient_id, profile).grid


@st.cache_data(show_spinner=False)
def get_benchmark(profile: str, model_mtime: float) -> dict:
    return get_engine(profile, model_mtime).benchmark_latency(n=25, warmup=3)


def checkpoint_mtime(profile: str) -> float:
    p = ROOT / "data" / "models" / f"biotwin_cmcl_{profile}.pt"
    return float(p.stat().st_mtime) if p.exists() else 0.0


# --------------------------------------------------------------------------- #
# Small view helpers
# --------------------------------------------------------------------------- #
def tier_chip(tier: str) -> str:
    color = TIER_COLORS.get(tier, MUTED)
    return (f'<span class="btx-chip" style="background:{color}22;color:{color};'
            f'border:1px solid {color}55">{tier}</span>')


def kpi(label: str, value: str, hint: str = "", color: str = TEXT) -> str:
    return (f'<div class="btx-kpi"><div class="label">{label}</div>'
            f'<div class="value" style="color:{color}">{value}</div>'
            f'<div class="hint">{hint}</div></div>')


def fmt_pct(x: float | None, digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{100 * float(x):.{digits}f}%"


URGENCY_COLOR = {
    "STAT": RED,
    "URGENT": "#fb923c",
    "PRIORITY": AMBER,
    "ROUTINE": ACCENT,
    "OPTIMISE": GREEN,
}


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def risk_gauge(risk: float, tier: str) -> go.Figure:
    color = TIER_COLORS.get(tier, ACCENT)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        number={"suffix": "", "font": {"size": 40, "color": TEXT}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"size": 10}},
            "bar": {"color": color, "thickness": 0.32},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 25], "color": "rgba(34,197,94,0.14)"},
                {"range": [25, 50], "color": "rgba(245,158,11,0.12)"},
                {"range": [50, 75], "color": "rgba(249,115,22,0.14)"},
                {"range": [75, 100], "color": "rgba(239,68,68,0.18)"},
            ],
            "threshold": {"line": {"color": color, "width": 3},
                          "thickness": 0.8, "value": risk},
        },
    ))
    fig.update_layout(**PLOT_LAYOUT, height=210)
    return fig


def event_bars(report: PatientReport) -> go.Figure:
    names = list(report.event_probs)
    vals = [report.event_probs[n] * 100 for n in names]
    labels = [n.replace("_", " ").title() for n in names]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=[ACCENT if report.dominant_event == n else "#3b4a66" for n in names],
                    line=dict(width=0)),
        text=[f"{v:.0f}%" for v in vals], textposition="outside",
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(**PLOT_LAYOUT, height=190, showlegend=False,
                      xaxis=dict(range=[0, 105], gridcolor=GRID, title="12-h probability"),
                      yaxis=dict(gridcolor="rgba(0,0,0,0)"))
    return fig


def risk_trend(series: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series["ts"], y=series["risk_index"], mode="lines",
        name="Clinical Risk Index", line=dict(color=ACCENT, width=2.4, shape="spline"),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.10)",
        hovertemplate="%{y:.1f}<extra></extra>",
    ))
    fig.add_hline(y=75, line=dict(color=RED, width=1, dash="dot"),
                  annotation_text="CRITICAL", annotation_font_color=RED)
    fig.add_hline(y=50, line=dict(color=AMBER, width=1, dash="dot"),
                  annotation_text="HIGH", annotation_font_color=AMBER)
    fig.update_layout(**PLOT_LAYOUT, height=240,
                      yaxis=dict(range=[0, 100], gridcolor=GRID, title="CRI"),
                      xaxis=dict(gridcolor=GRID))
    return fig


def physiology_chart(grid: pd.DataFrame, hours: float = 48) -> go.Figure:
    """Multi-metric, multi-axis live view of the asynchronous streams."""
    n = len(grid)
    span = int(hours * 60 / 5)
    g = grid.iloc[max(0, n - span):]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.07,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]],
        subplot_titles=("Autonomic: heart rate & RMSSD", "Glycaemic: CGM & time-in-range",
                        "Sleep architecture & signal coverage"),
    )

    fig.add_trace(go.Scatter(x=g["ts"], y=g["hr"], name="HR (bpm)",
                             line=dict(color="#f472b6", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=g["ts"], y=g["hrv_rmssd"], name="RMSSD (ms)",
                             line=dict(color=GREEN, width=1.5)),
                  row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=g["ts"], y=g["glucose"], name="Glucose (mg/dL)",
                             line=dict(color=VIOLET, width=1.6), fill="tozeroy",
                             fillcolor="rgba(167,139,250,0.08)"), row=2, col=1)
    fig.add_hline(y=70, line=dict(color=RED, width=1, dash="dot"), row=2, col=1)
    fig.add_hline(y=180, line=dict(color=AMBER, width=1, dash="dot"), row=2, col=1)
    fig.add_trace(go.Scatter(x=g["ts"], y=g["sleep_stage"], name="Sleep stage",
                             line=dict(color="#60a5fa", width=1.2, shape="hv"),
                             fill="tozeroy", fillcolor="rgba(96,165,250,0.10)"),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=g["ts"], y=g["coverage"] * 4, name="Coverage ×4",
                             line=dict(color=MUTED, width=1, dash="dash")), row=3, col=1)

    fig.update_layout(**PLOT_LAYOUT, height=620, showlegend=True)
    fig.update_yaxes(title_text="bpm", row=1, col=1, gridcolor=GRID)
    fig.update_yaxes(title_text="ms", row=1, col=1, secondary_y=True, gridcolor="rgba(0,0,0,0)")
    fig.update_yaxes(title_text="mg/dL", row=2, col=1, gridcolor=GRID)
    fig.update_yaxes(title_text="stage", row=3, col=1, gridcolor=GRID)
    for ann in fig.layout.annotations:
        ann.font.color = MUTED
        ann.font.size = 11
    return fig


def attribution_chart(explanation) -> go.Figure:
    atts = [a for a in explanation.attributions if a.value is not None]
    atts = sorted(atts, key=lambda a: -abs(a.value))[:9]
    fig = go.Figure(go.Bar(
        x=[a.value for a in atts][::-1],
        y=[a.feature.replace("_", " ") for a in atts][::-1],
        orientation="h",
        marker=dict(color=[GREEN if v >= 0 else RED for v in [a.value for a in atts][::-1]]),
        hovertemplate="%{y}: %{x:+.1f} CRI points<extra></extra>",
    ))
    fig.update_layout(**PLOT_LAYOUT, height=300, showlegend=False,
                      xaxis=dict(gridcolor=GRID, title="contribution to CRI"),
                      yaxis=dict(gridcolor="rgba(0,0,0,0)"))
    return fig


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def render_hero(engine: BioTwinEngine) -> None:
    backend = engine.backend
    backend_color = {"cmcl": GREEN, "surrogate": AMBER}.get(backend, RED)
    st.markdown(
        f'<div class="btx-hero">'
        f'<div class="mark">BioTwin<span>.X</span></div>'
        f'<div class="sub">Cross-modal contrastive digital twin · '
        f'predictive cardiometabolic prognosis</div>'
        f'<div style="flex:1"></div>'
        f'<span class="btx-chip" style="background:{backend_color}22;color:{backend_color};'
        f'border:1px solid {backend_color}55">backend · {backend}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_kpis(board: pd.DataFrame, engine: BioTwinEngine, latency: dict,
                profile: str) -> None:
    critical = int((board["tier"] == "CRITICAL").sum())
    high = int((board["tier"] == "HIGH").sum())
    elevated = int((board["tier"] == "ELEVATED").sum())
    imminent = int((board["pred_ttx_h"] <= 12).sum())
    metrics = engine.metrics or {}
    auc = metrics.get("cv", {}).get("any_event", {}).get("auc")
    auc_txt = f"{auc:.3f}" if isinstance(auc, (int, float)) and np.isfinite(auc) else "n/a"

    cols = st.columns(6)
    cards = [
        ("Patients monitored", f"{len(board)}", PROFILES[profile].describe(), TEXT),
        ("Critical", f"{critical}", "CRI ≥ 75", RED),
        ("High", f"{high}", "CRI 50–74", "#fb923c"),
        ("Elevated", f"{elevated}", "CRI 25–49", AMBER),
        ("Predicted < 12 h", f"{imminent}", "time-to-event", VIOLET),
        ("p95 latency", f"{latency.get('p95_ms', float('nan')):.0f} ms",
         f"SLO < {SLO_LATENCY_MS:.0f} ms", GREEN if latency.get("slo_met") else AMBER),
    ]
    for col, (label, value, hint, color) in zip(cols, cards):
        col.markdown(kpi(label, value, hint, color), unsafe_allow_html=True)

    st.caption(
        f"Model backend **{engine.backend}** · held-out any-event ROC-AUC **{auc_txt}** "
        f"· targets AUC ≥ {SLO_ROC_AUC:.2f}, F1 ≥ {SLO_F1:.2f}. "
        f"Metrics are measured on synthetic data; see the Model Card."
    )


def render_board(board: pd.DataFrame) -> None:
    st.markdown("### Triage board")
    st.caption("Ranked by Clinical Risk Index. Click a row, then open the patient tab.")

    view = board.copy()
    view["CRI"] = view["risk_index"].round(1)
    view["ischemia"] = (100 * view["p_silent_ischemia"]).round(0)
    view["crash"] = (100 * view["p_glycemic_crash"]).round(0)
    view["arrhythmia"] = (100 * view["p_autonomic_arrhythmia"]).round(0)
    view["TTX (h)"] = view["pred_ttx_h"].round(1)
    view["RMSSD dev"] = (100 * view["rmssd_dev_baseline"]).round(0)

    shown = view[[
        "patient_id", "tier", "CRI", "ischemia", "crash", "arrhythmia",
        "TTX (h)", "urgency", "RMSSD dev", "glu_min", "age", "sex", "prs_raw_z",
        "headline",
    ]]
    st.dataframe(
        shown,
        hide_index=True,
        width="stretch",
        height=min(560, 38 + 34 * len(shown)),
        column_config={
            "patient_id": st.column_config.TextColumn("Patient", width="small"),
            "tier": st.column_config.TextColumn("Tier", width="small"),
            "CRI": st.column_config.ProgressColumn(
                "CRI", min_value=0, max_value=100, format="%.1f"),
            "ischemia": st.column_config.NumberColumn("Ischemia %", format="%d%%"),
            "crash": st.column_config.NumberColumn("Crash %", format="%d%%"),
            "arrhythmia": st.column_config.NumberColumn("Arrhythmia %", format="%d%%"),
            "TTX (h)": st.column_config.NumberColumn("TTX (h)", format="%.1f"),
            "urgency": st.column_config.TextColumn("Urgency", width="small"),
            "RMSSD dev": st.column_config.NumberColumn("RMSSD Δ%", format="%d%%"),
            "glu_min": st.column_config.NumberColumn("Glucose min", format="%.0f"),
            "age": st.column_config.NumberColumn("Age", format="%d"),
            "sex": st.column_config.TextColumn("Sex", width="small"),
            "prs_raw_z": st.column_config.NumberColumn("PRS z", format="%.2f"),
            "headline": st.column_config.TextColumn("Primary action", width="large"),
        },
    )


def render_patient(engine: BioTwinEngine, board: pd.DataFrame, patient_id: str,
                   profile: str, model_mtime: float) -> None:
    report = engine.score_patient(patient_id)
    static = report.static
    tier_color = TIER_COLORS.get(report.tier, ACCENT)

    top = st.columns([1.05, 1.6, 1.35])
    with top[0]:
        st.markdown(
            f"#### {patient_id} &nbsp; {tier_chip(report.tier)}",
            unsafe_allow_html=True,
        )
        st.caption(
            f"{static.get('age', 0):.0f} y · {static.get('sex', '?')} · "
            f"BMI {static.get('bmi', 0):.1f} · HbA1c {static.get('hba1c', 0):.1f}%"
        )
        st.plotly_chart(risk_gauge(report.risk_index, report.tier),
                        width="stretch", config={"displayModeBar": False})

    with top[1]:
        st.markdown("##### 12-hour event probabilities")
        st.plotly_chart(event_bars(report), width="stretch",
                        config={"displayModeBar": False})
        st.caption(
            f"Dominant **{report.dominant_event.replace('_', ' ')}** · "
            f"predicted time-to-event **{report.predicted_ttx_hours:.1f} h** · "
            f"confidence **{fmt_pct(report.confidence)}**"
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Nocturnal RMSSD", f"{report.nocturnal_rmssd:.0f} ms"
                  if report.nocturnal_rmssd else "—")
        c2.metric("Personal baseline", f"{report.baseline_rmssd:.0f} ms"
                  if report.baseline_rmssd else "—")
        c3.metric("Signal coverage", fmt_pct(report.coverage))

    with top[2]:
        st.markdown("##### Genomic architecture")
        st.markdown(
            f"**Dominant variant** &nbsp; `{static.get('dominant_snp', '—')}`  \n"
            f"{static.get('dominant_gene', '—')} · dosage "
            f"**{static.get('dominant_dosage', 0)}/2** · "
            f"{static.get('dominant_trait', '')}"
        )
        st.markdown(
            f"**Polygenic burden**  \n"
            f"overall **{static.get('prs_raw_z', 0):+.2f}σ** · "
            f"CVD {static.get('prs_cvd_z', 0):+.2f}σ · "
            f"T2D {static.get('prs_t2d_z', 0):+.2f}σ · "
            f"autonomic {static.get('prs_aut_z', 0):+.2f}σ"
        )
        st.caption(
            "The genetic vector does not vote on the score directly. It "
            "modulates how the temporal encoder's evidence is read (FiLM), so "
            "the same physiology can mean different things in two patients."
        )

    # -- risk trend ------------------------------------------------------ #
    st.markdown("### Risk trajectory")
    series = get_series(profile, model_mtime, patient_id, 72.0)
    st.plotly_chart(risk_trend(series), width="stretch",
                    config={"displayModeBar": False})

    # -- physiology ------------------------------------------------------ #
    st.markdown("### Multi-modal telemetry")
    st.caption("Asynchronous streams on a common 5-minute analysis grid. "
               "Staleness channels preserve missingness instead of imputing it away.")
    grid = get_grid(profile, patient_id)
    st.plotly_chart(physiology_chart(grid), width="stretch")

    # -- actions --------------------------------------------------------- #
    st.markdown("### AI Actionable Insights Engine")
    if not report.actions:
        st.success("No action required. Continue routine monitoring.")
    for action in report.actions[:5]:
        color = URGENCY_COLOR.get(action.urgency, ACCENT)
        evidence = "".join(f"<li>{e}</li>" for e in action.evidence)
        st.markdown(
            f'<div class="btx-card" style="border-left:4px solid {color}">'
            f'<div class="urgency" style="color:{color}">{action.urgency}</div>'
            f'<div class="title">{action.title}</div>'
            f'<div class="rationale">{action.rationale}</div>'
            f'<ul>{evidence}</ul>'
            f'<div class="rationale" style="margin-top:8px;opacity:.75">'
            f'Source: {action.modality}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # -- explainability -------------------------------------------------- #
    with st.expander("Why this score? — attribution & narrative", expanded=False):
        if report.explanation:
            left, right = st.columns([1.2, 1])
            with left:
                st.plotly_chart(attribution_chart(report.explanation),
                                width="stretch", config={"displayModeBar": False})
            with right:
                st.markdown("**Modality share**")
                for group, share in sorted(report.explanation.modality_share.items(),
                                           key=lambda kv: -kv[1]):
                    st.progress(min(1.0, share), text=f"{group} · {fmt_pct(share)}")
                st.metric("Interaction share",
                          fmt_pct(report.explanation.interaction_share),
                          help="Risk explained by the *conjunction* of modalities, "
                               "not by any single signal.")
            st.info(report.explanation.narrative)

    with st.expander("Supporting observations", expanded=False):
        for ins in report.insights:
            st.markdown(f"**{ins.title}** — {ins.detail}  \n"
                        f"*{ins.severity} · score {ins.score:.0f}/100*")

    with st.expander("Data provenance & context vector", expanded=False):
        st.json({k: (round(float(v), 4) if isinstance(v, (int, float)) and np.isfinite(v) else v)
                 for k, v in report.context.items()})
        st.caption(
            f"As-of {report.as_of}. Backend `{report.backend}` · "
            f"scored in {report.latency_ms:.1f} ms. "
            f"Cohort `{profile}` · {PROFILES[profile].describe()}."
        )

    return report


def render_methodology(engine: BioTwinEngine, profile: str) -> None:
    st.markdown("### How the index is composed")
    st.latex(r"\mathrm{CRI} = 100 \times \left(0.55\,p_{\mathrm{event}}"
             r" + 0.25\,p_{\mathrm{class}} + 0.12\,\mathrm{strain}"
             r" + 0.08\,\sigma(\mathrm{PRS})\right)")
    st.markdown(
        "- **p_event** — calibrated 12-hour any-event probability (temperature-scaled\n"
        "  on held-out patients).\n"
        "- **p_class** — probability of the dominant event type; rewards specificity\n"
        "  over undifferentiated alarm.\n"
        "- **strain** — rule-based physiological strain from nocturnal RMSSD loss,\n"
        "  LF/HF withdrawal, hypoglycaemic minutes and glucose slope, attenuated by\n"
        "  signal coverage.\n"
        "- **σ(PRS)** — the polygenic prior, capped at 8 % so that genotype can shift\n"
        "  a score but never dominate measured physiology."
    )

    metrics = engine.metrics or {}
    cv = metrics.get("cv", {})
    if cv:
        st.markdown("#### Measured performance (patient-level CV)")
        rows = []
        for name, m in (cv.get("per_class") or {}).items():
            rows.append({"class": name, "ROC-AUC": m.get("auc"),
                         "F1": m.get("f1"), "AP": m.get("ap"),
                         "positives": m.get("n_pos")})
        if cv.get("any_event"):
            ae = cv["any_event"]
            rows.append({"class": "any event", "ROC-AUC": ae.get("auc"),
                         "F1": ae.get("f1"), "AP": ae.get("ap"),
                         "positives": ae.get("n_pos")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("These are the honest numbers from the last run in "
                   "`data/artifacts/`; the challenge targets are higher and are "
                   "tracked separately.")

    st.markdown("#### Ablations")
    abl = []
    for key, label in (("ablation_temporal_only", "Temporal encoder only"),
                       ("ablation_static_only", "Static/genomic only"),
                       ("baseline_xgboost_static", "XGBoost · static"),
                       ("baseline_xgboost_full", "XGBoost · static + summaries")):
        if key in metrics:
            m = metrics[key]
            abl.append({"variant": label,
                        "macro ROC-AUC": m.get("macro_auc_mean", m.get("auc")),
                        "any-event AUC": m.get("any_event_auc", m.get("auc"))})
    if abl:
        st.dataframe(pd.DataFrame(abl), hide_index=True, width="stretch")

    st.markdown("#### Streams ingested")
    st.dataframe(pd.DataFrame([{
        "modality": m.key, "unit": m.unit, "cadence (s)": m.cadence_s,
        "description": m.description} for m in MODALITIES]),
        hide_index=True, width="stretch")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title="BioTwin.X · Clinician Console",
        page_icon="🫀",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()

    with st.sidebar:
        st.markdown("### Console")
        profile = st.selectbox(
            "Cohort profile", sorted(PROFILES),
            index=sorted(PROFILES).index(DEFAULT_PROFILE) if DEFAULT_PROFILE in PROFILES else 0,
            help="demo = 20 patients × 7 days · standard = 100 × 14 · research = 240 × 30",
        )
        st.caption(PROFILES[profile].describe())
        allow_surrogate = st.toggle(
            "Allow surrogate fallback", value=True,
            help="If no trained checkpoint exists, fit a calibrated gradient-boosted "
                 "surrogate on the same feature space instead of showing nothing.")
        st.divider()
        st.markdown("### Filters")
        tier_filter = st.multiselect("Tiers", ["CRITICAL", "HIGH", "ELEVATED", "LOW"],
                                     default=["CRITICAL", "HIGH", "ELEVATED", "LOW"])
        min_risk = st.slider("Minimum CRI", 0, 100, 0, 5)
        st.divider()
        st.markdown("### Context")
        st.caption(
            "Synthetic cohort · research prototype\n\n"
            "Targets: AUC ≥ 0.94 · F1 ≥ 0.91 · p95 < 120 ms\n\n"
            "Not a medical device."
        )

    model_mtime = checkpoint_mtime(profile)
    with st.spinner("Assembling the digital twin — first run generates and caches "
                    "the cohort…"):
        engine = get_engine(profile, model_mtime, allow_surrogate)
        board = get_board(profile, model_mtime)

    render_hero(engine)

    if board.empty:
        st.warning("No patients scored. Generate a cohort with "
                   "`python inference_engine.py bootstrap`.")
        return

    latency = get_benchmark(profile, model_mtime)
    render_kpis(board, engine, latency, profile)

    filtered = board[
        board["tier"].isin(tier_filter) & (board["risk_index"] >= min_risk)
    ]
    if filtered.empty:
        st.info("No patients match the current filters.")
        filtered = board

    tabs = st.tabs(["🏥 Triage board", "🫀 Patient drill-down",
                    "🧠 Methodology & metrics", "ℹ️ About"])

    with tabs[0]:
        render_board(filtered)
        st.caption(
            "CRI is a triage aid, not a diagnosis. Every alert below carries its "
            "evidence chain so the clinician can overrule it."
        )
        st.markdown("#### Highest-priority handover")
        top_actions = filtered.head(3)
        for r in top_actions.itertuples(index=False):
            color = TIER_COLORS.get(r.tier, ACCENT)
            st.markdown(
                f'<div class="btx-card" style="border-left:4px solid {color}">'
                f'<div class="urgency" style="color:{color}">{r.urgency} · '
                f'CRI {r.risk_index:.1f} · {r.tier}</div>'
                f'<div class="title">{r.patient_id} — {r.headline}</div>'
                f'<div class="rationale">Predicted '
                f'{r.dominant_event.replace("_", " ")} in {r.pred_ttx_h:.1f} h. '
                f'Nocturnal RMSSD deviation {100 * r.rmssd_dev_baseline:+.0f}%, '
                f'glucose nadir {r.glu_min:.0f} mg/dL.</div>'
                f'</div>', unsafe_allow_html=True)

    with tabs[1]:
        default_id = str(filtered.iloc[0]["patient_id"])
        ids = filtered["patient_id"].tolist()
        fallback = ids.index(default_id) if default_id in ids else 0
        patient_id = st.selectbox("Patient", ids, index=fallback)
        render_patient(engine, filtered, patient_id, profile, model_mtime)

    with tabs[2]:
        render_methodology(engine, profile)

    with tabs[3]:
        st.markdown(
            "**BioTwin.X** fuses a static clinical + genomic architecture with "
            "asynchronous wearable telemetry through cross-modal contrastive "
            "learning. The temporal encoder reads 12 channels over a 12-hour "
            "window; the static encoder produces a patient archetype that "
            "*modulates* — never overrides — the temporal evidence (FiLM)."
        )
        st.markdown(
            "- **Data**: fully synthetic (`synthetic_generator.py`), generated "
            "with recurrent events and known ground truth.\n"
            "- **Model**: PatchTST-style transformer + NT-Xent alignment + "
            "supervised heads; calibrated with temperature scaling.\n"
            "- **Console**: Streamlit + Plotly.\n"
            "- **License**: MIT."
        )
        st.warning(
            "Research prototype. Trained on synthetic data only. Not validated "
            "on humans, not a medical device, and must not be used for clinical "
            "decisions."
        )

    st.markdown(
        '<div class="btx-disclaimer">'
        'BioTwin.X · Happiest Health Digital Twin Challenge 2026 · research '
        'prototype, synthetic data only · MIT License · '
        'Clinical Risk Index is a triage aid and does not constitute medical '
        'advice.'
        '</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
