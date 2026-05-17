import csv
import json
import os
import time
from typing import Dict, Optional

import torch


class RuntimeOverheadProfiler:
    """Record wall-time and CUDA-memory overhead for reviewer-facing analysis.

    Outputs under ``<model_path>/debug_overhead``:
      - runtime_metrics.csv: per-iteration wall time and overall memory.
      - consistency_overhead.csv: per-call multi-view consistency time/memory.
      - runtime_overhead_summary.txt: human-readable total/average runtime summary.
      - consistency_overhead.txt: append-only human-readable consistency overhead log.
      - memory_over_time.png / runtime_over_time.png / consistency_time.png / consistency_memory.png: visual diagnostics when matplotlib is available.
    """

    MB = 1024.0 * 1024.0

    def __init__(self, model_path: str, args):
        self.enabled = not getattr(args, "disable_overhead_profile", False)
        self.log_interval = max(1, int(getattr(args, "overhead_log_interval", 50)))
        self.plot_interval = max(0, int(getattr(args, "overhead_plot_interval", 500)))
        self.model_path = model_path
        self.start_time = time.perf_counter()
        self.iter_rows = []
        self.consistency_rows = []
        self.current_consistency_row = None

        self.debug_dir = os.path.join(model_path, "debug_overhead")
        self.runtime_csv = os.path.join(self.debug_dir, "runtime_metrics.csv")
        self.consistency_csv = os.path.join(self.debug_dir, "consistency_overhead.csv")
        self.summary_txt = os.path.join(self.debug_dir, "runtime_overhead_summary.txt")
        self.consistency_txt = os.path.join(self.debug_dir, "consistency_overhead.txt")

        if self.enabled:
            os.makedirs(self.debug_dir, exist_ok=True)
            self._write_header(self.runtime_csv, self.runtime_fields())
            self._write_header(self.consistency_csv, self.consistency_fields())
            with open(self.consistency_txt, "w") as f:
                f.write("# Multi-view consistency overhead log\n")

    @staticmethod
    def runtime_fields():
        return [
            "iteration",
            "wall_ms",
            "cuda_ms",
            "cuda_allocated_mb",
            "cuda_reserved_mb",
            "cuda_max_allocated_mb",
            "cuda_max_reserved_mb",
            "consistency_active",
            "consistency_ms",
            "consistency_peak_allocated_mb",
            "consistency_valid_ratio",
        ]

    @staticmethod
    def consistency_fields():
        return [
            "iteration",
            "elapsed_ms",
            "start_allocated_mb",
            "end_allocated_mb",
            "delta_allocated_mb",
            "start_reserved_mb",
            "end_reserved_mb",
            "delta_reserved_mb",
            "peak_allocated_mb",
            "peak_reserved_mb",
            "valid_ratio",
            "valid_pixels",
            "rough_l1",
            "metal_l1",
            "skipped",
        ]

    @staticmethod
    def _write_header(path: str, fieldnames) -> None:
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    @staticmethod
    def _sync_cuda() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _memory_stats(self) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {
                "allocated": 0.0,
                "reserved": 0.0,
                "max_allocated": 0.0,
                "max_reserved": 0.0,
            }
        return {
            "allocated": torch.cuda.memory_allocated() / self.MB,
            "reserved": torch.cuda.memory_reserved() / self.MB,
            "max_allocated": torch.cuda.max_memory_allocated() / self.MB,
            "max_reserved": torch.cuda.max_memory_reserved() / self.MB,
        }

    def start_iteration(self, iteration: int):
        if not self.enabled:
            return None
        self.current_consistency_row = None
        self._sync_cuda()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return {
            "iteration": iteration,
            "start_time": time.perf_counter(),
            "start_memory": self._memory_stats(),
        }

    def start_consistency(self, iteration: int):
        if not self.enabled:
            return None
        self._sync_cuda()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return {
            "iteration": iteration,
            "start_time": time.perf_counter(),
            "start_memory": self._memory_stats(),
        }

    def end_consistency(self, context, stats: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        if not self.enabled or context is None:
            return {}
        stats = stats or {}
        self._sync_cuda()
        end_memory = self._memory_stats()
        start_memory = context["start_memory"]
        row = {
            "iteration": context["iteration"],
            "elapsed_ms": (time.perf_counter() - context["start_time"]) * 1000.0,
            "start_allocated_mb": start_memory["allocated"],
            "end_allocated_mb": end_memory["allocated"],
            "delta_allocated_mb": end_memory["allocated"] - start_memory["allocated"],
            "start_reserved_mb": start_memory["reserved"],
            "end_reserved_mb": end_memory["reserved"],
            "delta_reserved_mb": end_memory["reserved"] - start_memory["reserved"],
            "peak_allocated_mb": end_memory["max_allocated"],
            "peak_reserved_mb": end_memory["max_reserved"],
            "valid_ratio": stats.get("valid_ratio", 0.0),
            "valid_pixels": stats.get("valid_pixels", 0.0),
            "rough_l1": stats.get("rough_l1", 0.0),
            "metal_l1": stats.get("metal_l1", 0.0),
            "skipped": stats.get("skipped", 1.0),
        }
        self.current_consistency_row = row
        self.consistency_rows.append(row)
        self._append_csv(self.consistency_csv, self.consistency_fields(), row)
        with open(self.consistency_txt, "a") as f:
            f.write(
                f"Iter {row['iteration']}: time={row['elapsed_ms']:.3f} ms, "
                f"alloc_delta={row['delta_allocated_mb']:.2f} MB, "
                f"peak_alloc={row['peak_allocated_mb']:.2f} MB, "
                f"reserved_delta={row['delta_reserved_mb']:.2f} MB, "
                f"valid_ratio={row['valid_ratio']:.5f}, skipped={row['skipped']}\n"
            )
        return row

    def record_iteration(self, context, iter_start_event=None, iter_end_event=None, consistency_active: bool = False) -> None:
        if not self.enabled or context is None:
            return
        self._sync_cuda()
        memory = self._memory_stats()
        cuda_ms = 0.0
        if iter_start_event is not None and iter_end_event is not None and torch.cuda.is_available():
            cuda_ms = float(iter_start_event.elapsed_time(iter_end_event))
        consistency_row = self.current_consistency_row or {}
        row = {
            "iteration": context["iteration"],
            "wall_ms": (time.perf_counter() - context["start_time"]) * 1000.0,
            "cuda_ms": cuda_ms,
            "cuda_allocated_mb": memory["allocated"],
            "cuda_reserved_mb": memory["reserved"],
            "cuda_max_allocated_mb": memory["max_allocated"],
            "cuda_max_reserved_mb": memory["max_reserved"],
            "consistency_active": int(consistency_active),
            "consistency_ms": consistency_row.get("elapsed_ms", 0.0),
            "consistency_peak_allocated_mb": consistency_row.get("peak_allocated_mb", 0.0),
            "consistency_valid_ratio": consistency_row.get("valid_ratio", 0.0),
        }
        self.iter_rows.append(row)
        if row["iteration"] % self.log_interval == 0 or consistency_active:
            self._append_csv(self.runtime_csv, self.runtime_fields(), row)
            self._write_summary(final=False)
        if self.plot_interval > 0 and row["iteration"] % self.plot_interval == 0:
            self.plot()

    @staticmethod
    def _append_csv(path: str, fieldnames, row: Dict[str, float]) -> None:
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

    def _write_summary(self, final: bool) -> None:
        total_s = time.perf_counter() - self.start_time
        avg_wall = sum(row["wall_ms"] for row in self.iter_rows) / max(1, len(self.iter_rows))
        avg_cuda = sum(row["cuda_ms"] for row in self.iter_rows) / max(1, len(self.iter_rows))
        avg_consistency = sum(row["elapsed_ms"] for row in self.consistency_rows) / max(1, len(self.consistency_rows))
        max_alloc = max([row["cuda_allocated_mb"] for row in self.iter_rows] or [0.0])
        max_reserved = max([row["cuda_reserved_mb"] for row in self.iter_rows] or [0.0])
        summary = {
            "final": final,
            "total_wall_seconds": total_s,
            "logged_iterations": len(self.iter_rows),
            "consistency_calls": len(self.consistency_rows),
            "avg_iteration_wall_ms": avg_wall,
            "avg_iteration_cuda_ms": avg_cuda,
            "avg_consistency_ms": avg_consistency,
            "max_cuda_allocated_mb": max_alloc,
            "max_cuda_reserved_mb": max_reserved,
        }
        with open(self.summary_txt, "w") as f:
            f.write("# Runtime and overhead summary\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        with open(os.path.join(self.debug_dir, "runtime_overhead_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    def plot(self) -> None:
        if not self.enabled:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            with open(self.summary_txt, "a") as f:
                f.write(f"plot_error: matplotlib unavailable ({exc})\n")
            return

        if self.iter_rows:
            xs = [row["iteration"] for row in self.iter_rows]
            plt.figure(figsize=(10, 5))
            plt.plot(xs, [row["cuda_allocated_mb"] for row in self.iter_rows], label="allocated MB")
            plt.plot(xs, [row["cuda_reserved_mb"] for row in self.iter_rows], label="reserved MB")
            plt.plot(xs, [row["cuda_max_allocated_mb"] for row in self.iter_rows], label="peak allocated MB", alpha=0.7)
            plt.xlabel("Iteration")
            plt.ylabel("CUDA memory (MB)")
            plt.title("Overall CUDA memory usage")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.debug_dir, "memory_over_time.png"), dpi=200)
            plt.close()

            plt.figure(figsize=(10, 5))
            plt.plot(xs, [row["wall_ms"] for row in self.iter_rows], label="wall ms")
            plt.plot(xs, [row["cuda_ms"] for row in self.iter_rows], label="cuda event ms")
            plt.xlabel("Iteration")
            plt.ylabel("Time (ms)")
            plt.title("Iteration runtime")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.debug_dir, "runtime_over_time.png"), dpi=200)
            plt.close()

        if self.consistency_rows:
            xs = [row["iteration"] for row in self.consistency_rows]
            plt.figure(figsize=(10, 5))
            plt.plot(xs, [row["elapsed_ms"] for row in self.consistency_rows], label="consistency time ms")
            plt.xlabel("Iteration")
            plt.ylabel("Time (ms)")
            plt.title("Multi-view consistency compute overhead")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.debug_dir, "consistency_time.png"), dpi=200)
            plt.close()

            plt.figure(figsize=(10, 5))
            plt.plot(xs, [row["peak_allocated_mb"] for row in self.consistency_rows], label="peak allocated MB")
            plt.plot(xs, [row["end_allocated_mb"] for row in self.consistency_rows], label="end allocated MB")
            plt.plot(xs, [row["delta_allocated_mb"] for row in self.consistency_rows], label="delta allocated MB")
            plt.xlabel("Iteration")
            plt.ylabel("CUDA memory (MB)")
            plt.title("Multi-view consistency memory overhead")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.debug_dir, "consistency_memory.png"), dpi=200)
            plt.close()

    def finalize(self) -> None:
        if not self.enabled:
            return
        self._sync_cuda()
        self._write_summary(final=True)
        self.plot()
