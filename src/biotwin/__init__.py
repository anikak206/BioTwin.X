"""
BioTwin.X — Next-Gen Multi-Modal Asynchronous Digital Twin
===========================================================
Cross-Modal Contrastive Learning (CMCL) for predictive cardiometabolic
prognosis: fusing static genomic/clinical architecture with high-frequency,
mutually asynchronous wearable telemetry.

Public surface
--------------
>>> from biotwin import BioTwinEngine
>>> engine = BioTwinEngine()
>>> report = engine.score_patient("BTX-0001")
>>> report.risk_index, report.tier
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = [
    "BioTwinEngine",
    "CohortBundle",
    "extract_features",
    "BioTwinCMCL",
    "nt_xent",
]

_LAZY = {
    "BioTwinEngine": ("biotwin.inference", "BioTwinEngine"),
    "CohortBundle": ("biotwin.dataio", "CohortBundle"),
    "extract_features": ("biotwin.features", "extract_features"),
    "BioTwinCMCL": ("biotwin.cmcl", "BioTwinCMCL"),
    "nt_xent": ("biotwin.cmcl", "nt_xent"),
}


def __getattr__(name: str):  # PEP 562 lazy attribute access
    if name in _LAZY:
        import importlib
        mod, attr = _LAZY[name]
        return getattr(importlib.import_module(mod), attr)
    raise AttributeError(f"module 'biotwin' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + list(_LAZY.keys())))
