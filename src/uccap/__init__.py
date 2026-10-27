"""uccap: process capability analysis, risk-calibrated approval and hybrid statistical learning
decision support with finite samples.

Reference implementation of Jiang & Yang (IJAMT, 2026):
  * uccap.workflow  Practical process capability indices workflows, doi:10.1007/s00170-026-17782-7
  * uccap.risk      Risk-calibrated process capability approval with finite samples, doi:10.1007/s00170-026-18284-2
  * uccap.hybrid    A hybrid statistical learning framework for capability-based decision support
                    in manufacturing, doi:10.1007/s00170-026-18833-9
  * uccap.monitor   Bayesian operational acceptability monitoring for manufacturing process control
                    (accepted)
"""
from ._cite import cite
from .batch import BatchResult, batch_study
from .hybrid import HybridRisk, hybrid_risk
from .io import TableInput, read_clipboard, read_data, read_table, read_text
from .monitor import MonitorBatchResult, MonitorResult, batch_monitor, monitor_study
from .risk import RiskDecision, SamplePlan, risk_decision, sample_size
from .workflow import CapabilityStudy, Spec, capability_study

__version__ = "0.1.0"
__all__ = ["capability_study", "CapabilityStudy", "Spec", "risk_decision", "RiskDecision", "sample_size", "SamplePlan",
           "hybrid_risk", "HybridRisk", "batch_study", "BatchResult", "read_table", "TableInput", "read_data", "read_text",
           "read_clipboard", "monitor_study", "MonitorResult", "batch_monitor", "MonitorBatchResult", "cite"]
