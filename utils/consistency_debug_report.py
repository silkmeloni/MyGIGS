#!/usr/bin/env python3
"""Summarize multi-view material-consistency diagnostics and suggest next sweeps."""

import argparse
import csv
import json
import os
from statistics import mean


def _to_float(row, key, default=0.0):
    value = row.get(key, "")
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def load_rows(debug_dir):
    csv_path = os.path.join(debug_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"metrics.csv not found under {debug_dir}")
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def summarize(rows):
    active = [r for r in rows if _to_float(r, "weighted_loss") > 0]
    target = active or rows
    summary = {
        "num_rows": len(rows),
        "num_weighted_rows": len(active),
        "skip_rate": mean([_to_float(r, "skipped") for r in rows]) if rows else None,
        "valid_ratio_mean": mean([_to_float(r, "valid_ratio") for r in target]) if target else None,
        "occ_ratio_mean": mean([_to_float(r, "occ_ratio") for r in target]) if target else None,
        "edge_ratio_mean": mean([_to_float(r, "edge_ratio") for r in target]) if target else None,
        "facing_ratio_mean": mean([_to_float(r, "facing_ratio") for r in target]) if target else None,
        "rough_l1_mean": mean([_to_float(r, "rough_l1") for r in target]) if target else None,
        "metal_l1_mean": mean([_to_float(r, "metal_l1") for r in target]) if target else None,
        "latest_recommendation": rows[-1].get("recommendation", "") if rows else "",
    }
    return summary


def suggest(summary):
    suggestions = []
    if summary["num_rows"] == 0:
        suggestions.append("No consistency diagnostics found; check --use_consistency and --consistency_log_interval.")
        return suggestions
    if summary["num_weighted_rows"] == 0:
        suggestions.append("All consistency samples were skipped: inspect debug_consistency images; try closer views with --consistency_rank_max 2 or relax --consistency_min_valid_ratio.")
    if summary["valid_ratio_mean"] is not None and summary["valid_ratio_mean"] < 0.01:
        suggestions.append("Very low valid overlap: use closer target views, relax occlusion/edge thresholds, or delay consistency until geometry is stable.")
    if summary["occ_ratio_mean"] is not None and summary["occ_ratio_mean"] < 0.2:
        suggestions.append("Occlusion mask rejects most pixels: verify projection/depth scale and consider increasing --consistency_occ_abs_thresh or --consistency_occ_rel_thresh.")
    if summary["edge_ratio_mean"] is not None and summary["edge_ratio_mean"] < 0.2:
        suggestions.append("Depth-edge mask is too strict: increase --consistency_edge_rel_thresh if valid surfaces are black in the edge mask.")
    disagreement = (summary.get("rough_l1_mean") or 0.0) + (summary.get("metal_l1_mean") or 0.0)
    if disagreement > 0.5:
        suggestions.append("Large material disagreement: lower --lambda_consistency, increase --consistency_start_offset, or keep --consistency_albedo disabled.")
    if not suggestions:
        suggestions.append("Consistency diagnostics look usable; sweep --lambda_consistency in {0.005,0.01,0.02,0.05} and compare PSNR/material smoothness.")
    return suggestions


def format_report(summary, suggestions):
    lines = ["# Multi-view Material Consistency Debug Report", "", "## Metrics"]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Suggested next iterations")
    for suggestion in suggestions:
        lines.append(f"- {suggestion}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default=None, help="Training output directory containing debug_consistency/.")
    parser.add_argument("--debug_dir", default=None, help="Direct path to debug_consistency/.")
    parser.add_argument("--write_report", action="store_true", help="Write report.md into the debug directory.")
    args = parser.parse_args()

    debug_dir = args.debug_dir
    if debug_dir is None:
        if args.model_path is None:
            raise SystemExit("Pass --model_path or --debug_dir")
        debug_dir = os.path.join(args.model_path, "debug_consistency")

    rows = load_rows(debug_dir)
    summary = summarize(rows)
    suggestions = suggest(summary)
    report = format_report(summary, suggestions)
    print(report)

    with open(os.path.join(debug_dir, "offline_summary.json"), "w") as f:
        json.dump({"summary": summary, "suggestions": suggestions}, f, indent=2)
    if args.write_report:
        with open(os.path.join(debug_dir, "report.md"), "w") as f:
            f.write(report)


if __name__ == "__main__":
    main()
