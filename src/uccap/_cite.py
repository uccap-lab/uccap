"""BibTeX entries of the articles behind each module."""
from __future__ import annotations

from typing import Optional

ENTRIES = {
    "workflow": """@article{jiang2026practical,
  author = {Jiang, Fei and Yang, Lei},
  title = {Practical Process Capability Indices Workflows},
  journal = {The International Journal of Advanced Manufacturing Technology},
  year = {2026}, volume = {143}, number = {7--8}, pages = {3581--3599},
  publisher = {Springer}, doi = {10.1007/s00170-026-17782-7}
}""",
    "risk": """@article{jiang2026risk,
  author = {Jiang, Fei and Yang, Lei},
  title = {Risk-Calibrated Process Capability Approval with Finite Samples},
  journal = {The International Journal of Advanced Manufacturing Technology},
  year = {2026}, volume = {144}, number = {9--10}, pages = {7189--7206},
  publisher = {Springer}, doi = {10.1007/s00170-026-18284-2}
}""",
    "hybrid": """@article{jiang2026hybrid,
  author = {Jiang, Fei and Yang, Lei},
  title = {A Hybrid Statistical Learning Framework for Capability-Based Decision Support in Manufacturing},
  journal = {The International Journal of Advanced Manufacturing Technology},
  year = {2026}, publisher = {Springer}, doi = {10.1007/s00170-026-18833-9}
}""",
    "monitor": """@article{jiang2026bayesian,
  author = {Jiang, Fei and Yang, Lei},
  title = {Bayesian Operational Acceptability Monitoring for Manufacturing Process Control},
  journal = {The International Journal of Advanced Manufacturing Technology},
  year = {2026}, publisher = {Springer}, note = {Accepted}
}""",
}

MODULES = {"workflow": "uccap.workflow, capability_study", "risk": "uccap.risk, risk_decision, sample_size",
           "hybrid": "uccap.hybrid, hybrid_risk", "monitor": "uccap.monitor, monitor_study, batch_monitor"}


def cite(module: Optional[str] = None) -> str:
    """BibTeX of the article behind ``module`` ("workflow", "risk", "hybrid" or "monitor"), or of all four.

    ``batch_study`` uses the first three modules; ``batch_monitor`` uses the fourth.
    """
    keys = list(ENTRIES) if module is None else [module]
    unknown = [k for k in keys if k not in ENTRIES]
    if unknown:
        raise ValueError(f"module must be one of {sorted(ENTRIES)}")
    parts = [f"% {MODULES[k]}\n{ENTRIES[k]}" for k in keys]
    return "\n\n".join(parts)
