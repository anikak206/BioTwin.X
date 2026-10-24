#!/usr/bin/env python3
"""
BioTwin.X — Clinical Inference Engine (reference CLI)
=====================================================
This is the *operational* entry point to the digital twin. It wraps
:class:`biotwin.inference.BioTwinEngine` so that a clinician, a reviewer or a
CI job can drive the platform without opening the Streamlit console.

Subcommands
-----------
``bootstrap``   Generate the cohort, extract features and warm every cache.
``report``      Full single-patient report: risk index, evidence, actions.
``cohort``      The triage board — every patient ranked by Clinical Risk Index.
``series``      The risk trajectory over the trailing window (CSV / table).
``benchmark``   End-to-end latency percentiles against the 120 ms SLO.
``doctor``      Doctor-facing summary of the top-N patients, one screen each.

Examples
--------
    python inference_engine.py bootstrap --profile demo
    python inference_engine.py report --profile demo --patient BTX-0004
    python inference_engine.py cohort --profile standard --top 15 --json out.json
    python inference_engine.py benchmark --profile demo --samples 80
    python inference_engine.py series --profile demo --patient BTX-0004 --hours 48

Design notes
------------
* No hidden state: every command is a pure function of the cohort on disk.
* The engine degrades in *accuracy*, never in *availability*. If no trained
  checkpoint exists, a calibrated gradient-boosted surrogate is fitted on the
  same feature space; if even that is impossible, a deterministic rule engine
  answers. The active backend is always printed, never silently swapped.
* Exit codes: ``0`` success, ``2`` bad input, ``3`` no data / could not score.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from biotwin.config import (  # noqa: E402
    DEFAULT_PROFILE,
    PROFILES,
    SLO_F1,
    SLO_LATENCY_MS,
    SLO_ROC_AUC,
)
from biotwin.inference import BioTwinEngine, PatientReport  # noqa: E402
from biotwin.risk import TIER_COLORS  # noqa: E402

# --------------------------------------------------------------------------- #
# Terminal presentation
# --------------------------------------------------------------------------- #
_NO_COLOR = bool(os.getenv("NO_COLOR")) or not sys.stdout.isatty()

_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}
_TIER_ANSI = {
    "LOW": "green",
    "ELEVATED": "yellow",
    "HIGH": "yellow",
    "CRITICAL": "red",
}
_SEVERITY_MARK = {
    "critical": "!!",
    "warning": " !",
    "info": " ·",
    "reassuring": " ✓",
}


def c(text: Any, *styles: str) -> str:
    """Colourise a string when the terminal supports it (NO_COLOR respected)."""
    s = str(text)
    if _NO_COLOR or not styles:
        return s
    return "".join(_ANSI.get(k, "") for k in styles) + s + _ANSI["reset"]


def rule(char: str = "─", width: int = 78) -> str:
    return char * width


def say(msg: str = "") -> None:
    print(msg, flush=True)


def header(title: str) -> None:
    say()
    say(c(rule("═"), "dim"))
    say(c(f"  BioTwin.X · {title}", "bold", "cyan"))
    say(c(rule("═"), "dim"))


def kv(label: str, value: Any, style: str | None = None) -> None:
    text = str(value)
    if style:
        text = c(text, style)
    say(f"  {c(label.ljust(26), 'dim')}{text}")


def pct(x: float, digits: int = 1) -> str:
    return f"{100.0 * float(x):.{digits}f}%"


# --------------------------------------------------------------------------- #
# Engine construction
# --------------------------------------------------------------------------- #
def make_engine(args: argparse.Namespace) -> BioTwinEngine:
    """Build the engine, explaining the backend choice to the operator."""
    if args.profile not in PROFILES:
        say(c(f"unknown profile '{args.profile}'", "red"))
        say(f"available: {', '.join(sorted(PROFILES))}")
        raise SystemExit(2)
    if not args.quiet:
        say(c(f"▸ profile {args.profile} — {PROFILES[args.profile].describe()}", "dim"))
    engine = BioTwinEngine(
        profile=args.profile,
        model_path=Path(args.model) if getattr(args, "model", None) else None,
        allow_surrogate=not getattr(args, "no_surrogate", False),
        verbose=not args.quiet,
    )
    if not args.quiet:
        say(c(f"▸ backend {engine.backend}", "dim"))
    return engine


def check_patient(engine: BioTwinEngine, patient_id: str | None) -> str:
    ids = engine.bundle.patient_ids()
    if not ids:
        say(c("cohort is empty — run `bootstrap` first", "red"))
        raise SystemExit(3)
    if patient_id is None:
        board = engine.score_cohort()
        return str(board.iloc[0]["patient_id"])
    if patient_id not in ids:
        say(c(f"patient '{patient_id}' is not in cohort {engine.profile.name}", "red"))
        say(f"first ids: {', '.join(ids[:8])}")
        raise SystemExit(2)
    return patient_id


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_bootstrap(args: argparse.Namespace) -> int:
    from biotwin.dataio import CohortBundle, build_tensors, patient_features

    header("bootstrap")
    bundle = CohortBundle.load(args.profile, force_regen=args.force)
    ids = bundle.patient_ids()
    kv("patients", len(ids))
    kv("events", len(bundle.events))
    kv("raw telemetry", str(Path("data/raw") / f"{args.profile}_*.parquet"))

    import time

    t0 = time.perf_counter()
    for i, pid in enumerate(ids, 1):
        patient_features(pid, bundle.profile)
        if args.verbose or i == len(ids) or i % 10 == 0:
            say(f"  … features {i}/{len(ids)} ({time.perf_counter() - t0:.1f}s)")
    kv("feature grids", f"cached in {time.perf_counter() - t0:.1f}s")

    if not args.skip_tensors:
        t1 = time.perf_counter()
        tensors = build_tensors(bundle, stride_bins=args.stride)
        kv("training windows", f"{len(tensors):,} ({time.perf_counter() - t1:.1f}s)")
    say()
    say(c("  ready — try:  python inference_engine.py report --patient " + ids[0], "green"))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    engine = make_engine(args)
    pid = check_patient(engine, args.patient)
    rep = engine.score_patient(pid)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep.to_dict(), indent=2, default=str), encoding="utf-8")
        say(c(f"▸ wrote {out}", "green"))
        return 0

    render_report(rep, detail=not args.brief, show_context=args.context)
    return 0


def render_report(rep: PatientReport, detail: bool = True,
                  show_context: bool = False) -> None:
    """Human-readable single-patient report (also used by ``doctor``)."""
    tier = rep.tier
    tier_style = _TIER_ANSI.get(tier, "bold")
    header(f"patient {rep.patient_id}")

    say(f"  {c('CLINICAL RISK INDEX', 'dim')}   "
        f"{c(f'{rep.risk_index:5.1f} / 100', 'bold', tier_style)}   "
        f"{c(tier, 'bold', tier_style)}")
    bar_len = 42
    filled = int(round(bar_len * rep.risk_index / 100.0))
    bar = "█" * filled + "░" * (bar_len - filled)
    say(f"  {c(bar, tier_style)}")
    say()

    s = rep.static
    kv("age / sex / BMI",
       f"{s.get('age', float('nan')):.0f} y · {s.get('sex', '?')} · "
       f"{s.get('bmi', float('nan')):.1f} kg/m²")
    kv("HbA1c / LDL / SBP",
       f"{s.get('hba1c', float('nan')):.1f}% · {s.get('ldl', float('nan')):.0f} mg/dL · "
       f"{s.get('systolic_bp', float('nan')):.0f} mmHg")
    kv("polygenic risk",
       f"overall z {s.get('prs_raw_z', 0.0):+.2f}σ · CVD z {s.get('prs_cvd_z', 0.0):+.2f}σ · "
       f"T2D z {s.get('prs_t2d_z', 0.0):+.2f}σ")
    kv("dominant variant",
       f"{s.get('dominant_snp', '—')} ({s.get('dominant_gene', '—')}) "
       f"dosage {s.get('dominant_dosage', 0)}/2 · {s.get('dominant_trait', '')}")
    kv("as-of", rep.as_of.strftime("%Y-%m-%d %H:%M"))
    kv("confidence / coverage", f"{pct(rep.confidence)} · {pct(rep.coverage)}")
    kv("backend / latency", f"{rep.backend} · {rep.latency_ms:.1f} ms")

    say()
    say(c("  EVENT PROBABILITIES (12-hour horizon)", "bold"))
    for name, p in sorted(rep.event_probs.items(), key=lambda kv_: -kv_[1]):
        mark = "◀ dominant" if name == rep.dominant_event else ""
        say(f"    {name.ljust(24)} {c(pct(p), 'bold')}  {c(mark, 'dim')}")
    kv("predicted time-to-event", f"{rep.predicted_ttx_hours:.1f} h")

    say()
    say(c("  AI ACTIONABLE INSIGHTS ENGINE", "bold", "magenta"))
    if not rep.actions:
        say(c("    No action required — continue routine monitoring.", "green"))
    else:
        for i, a in enumerate(rep.actions[: 6 if detail else 2], 1):
            urgency_style = {"STAT": "red", "URGENT": "yellow",
                             "PRIORITY": "yellow"}.get(a.urgency, "cyan")
            say(f"  {i}. {c(f'[{a.urgency}]', 'bold', urgency_style)} "
                f"{c(a.title, 'bold')}")
            say(f"     {c(a.rationale, 'dim')}")
            if detail:
                for e in a.evidence:
                    say(f"       · {e}")
                if a.modality:
                    say(f"       {c('source: ' + a.modality, 'dim')}")

    if detail and rep.insights:
        say()
        say(c("  SUPPORTING OBSERVATIONS", "bold"))
        for ins in rep.insights[:6]:
            mark = _SEVERITY_MARK.get(ins.severity, " ·")
            say(f"    {mark} {ins.title} {c(f'({ins.score:.0f})', 'dim')}")
            say(f"       {c(ins.detail, 'dim')}")

    if detail and rep.explanation:
        say()
        say(c("  WHY THIS SCORE — modality attribution", "bold"))
        for group, share in sorted(rep.explanation.modality_share.items(),
                                   key=lambda kv_: -kv_[1]):
            bar = "▪" * max(1, int(round(share * 24)))
            say(f"    {group.ljust(20)} {pct(share, 0):>5}  {c(bar, 'cyan')}")
        kv("interaction share", pct(rep.explanation.interaction_share))
        say()
        say(c("  " + rep.explanation.narrative, "italic" if False else "dim"))

    if show_context:
        say()
        say(c("  ENGINEERED CONTEXT (raw)", "bold"))
        for k, v in sorted(rep.context.items()):
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                continue
            say(f"    {k.ljust(30)} {v:>10.3f}")

    say()
    say(c(rule(), "dim"))


def cmd_cohort(args: argparse.Namespace) -> int:
    engine = make_engine(args)
    board = engine.score_cohort()
    board.to_csv(args.csv) if args.csv else None

    if args.json:
        payload = {
            "profile": engine.profile.name,
            "backend": engine.backend,
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "patients": json.loads(board.to_json(orient="records", date_format="iso")),
        }
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        say(c(f"▸ wrote {args.json}", "green"))

    header(f"triage board · {engine.profile.name} · {len(board)} patients")
    cols = ["patient_id", "risk_index", "tier", "p_silent_ischemia",
            "p_glycemic_crash", "p_autonomic_arrhythmia", "pred_ttx_h", "urgency"]
    view = board.head(args.top) if args.top else board

    say(f"  {'PATIENT':<10}{'CRI':>6}  {'TIER':<9}{'ISCHEMIA':>9}"
        f"{'CRASH':>8}{'ARRHYTH':>9}{'TTX(h)':>8}  URGENCY")
    say(c("  " + rule("-", 76), "dim"))
    for r in view[cols].itertuples(index=False):
        tier_style = _TIER_ANSI.get(r.tier, "reset")
        say(f"  {r.patient_id:<10}{r.risk_index:>6.1f}  "
            f"{c(r.tier.ljust(9), tier_style)}"
            f"{100 * r.p_silent_ischemia:>8.0f}%"
            f"{100 * r.p_glycemic_crash:>7.0f}%"
            f"{100 * r.p_autonomic_arrhythmia:>8.0f}%"
            f"{r.pred_ttx_h:>8.1f}  {r.urgency}")
    say(c("  " + rule("-", 76), "dim"))

    counts = board["tier"].value_counts().to_dict()
    say(f"  tiers: " + " · ".join(
        f"{c(k, _TIER_ANSI.get(k, 'reset'))} {counts.get(k, 0)}"
        for k in ("CRITICAL", "HIGH", "ELEVATED", "LOW")))
    say(f"  backend {engine.backend} · median latency "
        f"{board['latency_ms'].median():.1f} ms")
    if args.top:
        say(c(f"  showing top {min(args.top, len(board))} of {len(board)}", "dim"))
    return 0


def cmd_series(args: argparse.Namespace) -> int:
    engine = make_engine(args)
    pid = check_patient(engine, args.patient)
    series = engine.risk_series(pid, hours=args.hours, step_minutes=args.step)

    if args.csv:
        series.to_csv(args.csv, index=False)
        say(c(f"▸ wrote {args.csv}", "green"))
        return 0

    header(f"risk trajectory · {pid} · last {args.hours:.0f} h")
    say(f"  {'TIMESTAMP':<18}{'CRI':>7}{'RMSSD':>8}{'HR':>7}{'GLU':>7}{'LF/HF':>8}  EVENT")
    say(c("  " + rule("-", 74), "dim"))
    for r in series.itertuples(index=False):
        ts = pd.Timestamp(r.ts).strftime("%Y-%m-%d %H:%M")
        flag = ""
        if r.risk_index >= 75:
            flag = c("CRITICAL", "red")
        elif r.risk_index >= 50:
            flag = c("HIGH", "yellow")
        say(f"  {ts:<18}{r.risk_index:>7.1f}{_num(r.hrv_rmssd):>8}"
            f"{_num(r.hr):>7}{_num(r.glucose):>7}{_num(r.lf_hf):>8}  {flag}")
    say(c("  " + rule("-", 74), "dim"))
    say(f"  peak CRI {series['risk_index'].max():.1f} · "
        f"mean {series['risk_index'].mean():.1f} · "
        f"{len(series)} points at {args.step:g}-min resolution")
    return 0


def _num(v: Any, digits: int = 1) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return f"{f:.{digits}f}" if np.isfinite(f) else "—"


def cmd_benchmark(args: argparse.Namespace) -> int:
    engine = make_engine(args)
    if not engine.bundle.patient_ids():
        say(c("cohort is empty — run `bootstrap` first", "red"))
        return 3
    stats = engine.benchmark_latency(n=args.samples, warmup=args.warmup)
    header("latency benchmark")
    kv("backend", engine.backend)
    kv("samples", args.samples)
    for k in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"):
        kv(k.replace("_", " "), f"{stats[k]:.2f} ms")
    verdict = ("PASS" if stats["slo_met"] else "FAIL")
    style = "green" if stats["slo_met"] else "red"
    kv("SLO (p95 < 120 ms)", c(verdict, "bold", style))
    say()
    say(c("  Note: each sample re-assembles the 12-hour window and runs the "
          "full explanation + action engine,", "dim"))
    say(c("  i.e. this is the doctor-facing latency, not the bare forward pass.",
          "dim"))
    if args.json:
        Path(args.json).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    engine = make_engine(args)
    board = engine.score_cohort()
    n = min(args.top, len(board))
    header(f"doctor handover · top {n} by Clinical Risk Index")
    for pid in board.head(n)["patient_id"]:
        render_report(engine.score_patient(pid), detail=False, show_context=False)
    return 0


def cmd_targets(_: argparse.Namespace) -> int:
    header("advertised SLOs vs. measurement protocol")
    kv("ROC-AUC target", f"≥ {SLO_ROC_AUC:.2f} (macro, patient-level 5-fold CV)")
    kv("F1 target", f"≥ {SLO_F1:.2f} (event class, tuned threshold)")
    kv("p95 latency target", f"< {SLO_LATENCY_MS:.0f} ms end-to-end")
    say()
    say("  These are challenge *targets*. Measured values live in "
        "data/artifacts/training_report_*.json")
    say("  and in README §6.5. The pipeline never fabricates a metric: if the "
        "target is missed,")
    say("  the report says so.")
    return 0


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="inference_engine.py",
        description="BioTwin.X — cross-modal contrastive digital twin (CLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Design notes")[0],
    )
    ap.add_argument("--profile", default=DEFAULT_PROFILE, choices=sorted(PROFILES),
                    help="cohort profile (default: %(default)s)")
    ap.add_argument("--model", default=None,
                    help="path to a CMCL checkpoint (default: data/models/…)")
    ap.add_argument("--quiet", action="store_true", help="suppress progress output")
    ap.add_argument("--no-surrogate", action="store_true",
                    help="never fit the fallback surrogate (rules only)")

    sub = ap.add_subparsers(dest="command", required=True)

    b = sub.add_parser("bootstrap", help="generate cohort + warm caches")
    b.add_argument("--force", action="store_true", help="regenerate the cohort")
    b.add_argument("--skip-tensors", action="store_true")
    b.add_argument("--stride", type=int, default=6, help="window stride in 5-min bins")
    b.add_argument("--verbose", action="store_true")
    b.set_defaults(func=cmd_bootstrap)

    r = sub.add_parser("report", help="single-patient report")
    r.add_argument("--patient", default=None, help="default: highest-risk patient")
    r.add_argument("--json", default=None, help="write the full report as JSON")
    r.add_argument("--brief", action="store_true", help="hide evidence + narrative")
    r.add_argument("--context", action="store_true", help="dump raw context features")
    r.set_defaults(func=cmd_report)

    co = sub.add_parser("cohort", help="triage board")
    co.add_argument("--top", type=int, default=25)
    co.add_argument("--json", default=None)
    co.add_argument("--csv", default=None)
    co.set_defaults(func=cmd_cohort)

    s = sub.add_parser("series", help="risk trajectory for one patient")
    s.add_argument("--patient", default=None)
    s.add_argument("--hours", type=float, default=48.0)
    s.add_argument("--step", type=float, default=30.0, help="minutes per point")
    s.add_argument("--csv", default=None)
    s.set_defaults(func=cmd_series)

    bm = sub.add_parser("benchmark", help="latency percentiles vs. SLO")
    bm.add_argument("--samples", type=int, default=40)
    bm.add_argument("--warmup", type=int, default=5)
    bm.add_argument("--json", default=None)
    bm.set_defaults(func=cmd_benchmark)

    d = sub.add_parser("doctor", help="top-N handover sheets")
    d.add_argument("--top", type=int, default=3)
    d.set_defaults(func=cmd_doctor)

    t = sub.add_parser("targets", help="show the advertised SLOs")
    t.set_defaults(func=cmd_targets)

    return ap


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args) or 0)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        say(c("\ninterrupted", "yellow"))
        return 130
    except FileNotFoundError as exc:
        say(c(f"missing file: {exc}", "red"))
        say("hint: run `python inference_engine.py bootstrap --profile "
            f"{args.profile}` first")
        return 3
    except Exception as exc:  # pragma: no cover - last-resort guard
        say(c(f"{type(exc).__name__}: {exc}", "red"))
        if os.getenv("BIOTWIN_DEBUG"):
            raise
        say(c("set BIOTWIN_DEBUG=1 for a traceback", "dim"))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
