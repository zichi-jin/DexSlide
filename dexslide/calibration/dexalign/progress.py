"""Progress logging adapter for SciPy optimizers."""

from __future__ import annotations

import time
from typing import Callable

from scipy.optimize import OptimizeResult
import numpy as np


class SolverProgressLogger:
    def __init__(self, stage_name: str, *, log_fn: Callable[[str], None] | None, min_interval_sec: float = 1.0, metric_label: str | None = None, metric_getter: Callable[[], float | None] | None = None) -> None:
        self.stage_name = str(stage_name)
        self.log_fn = log_fn
        self.min_interval_sec = float(min_interval_sec)
        self.metric_label = None if metric_label is None else str(metric_label)
        self.metric_getter = metric_getter
        self.start_time = time.monotonic()
        self.last_log_time = self.start_time
        self.eval_count = 0
        self.best_cost = float("inf")

    def wrap(self, residual_fn: Callable) -> Callable:
        def wrapped(params):
            residual = np.asarray(residual_fn(params), dtype=np.float64).reshape(-1)
            self.eval_count += 1
            finite = residual[np.isfinite(residual)]
            cost = float("inf") if finite.size == 0 else float(0.5 * np.dot(finite, finite))
            self.best_cost = min(self.best_cost, cost)
            now = time.monotonic()
            if self.log_fn is not None and now - self.last_log_time >= self.min_interval_sec:
                rms = float("nan") if finite.size == 0 else float(np.sqrt(np.mean(np.square(finite))))
                metric_suffix = ""
                if self.metric_label is not None and self.metric_getter is not None:
                    metric_value = self.metric_getter()
                    if metric_value is not None and np.isfinite(float(metric_value)):
                        metric_suffix = f" {self.metric_label}={float(metric_value):.6g}"
                self.log_fn(f"[dexalign2] {self.stage_name}: evals={self.eval_count} cost={cost:.6g} best={self.best_cost:.6g} weighted_rms={rms:.6g}{metric_suffix} elapsed={now - self.start_time:.1f}s")
                self.last_log_time = now
            return residual

        return wrapped

    def finish(self, result: OptimizeResult) -> None:
        if self.log_fn is None:
            return
        elapsed = time.monotonic() - self.start_time
        self.log_fn(f"[dexalign2] {self.stage_name}: done success={bool(result.success)} status={int(result.status)} nfev={int(result.nfev)} cost={float(result.cost):.6g} elapsed={elapsed:.1f}s")
