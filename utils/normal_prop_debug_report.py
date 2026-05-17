#!/usr/bin/env python3
"""Summarize normal-propagation file diagnostics and suggest next sweep settings."""

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
    active = [r for r in rows if int(float(r.get("active", 0))) == 1]
    reference = [r for r in rows if int(float(r.get("active", 0))) == 0]
    target = active or rows

    summary = {
        "num_rows": len(rows),
        "num_active_rows": len(active),
        "reference_psnr_mean": mean([_to_float(r, "train_psnr") for r in reference]) if reference else None,
        "active_psnr_mean": mean([_to_float(r, "train_psnr") for r in active]) if active else None,
        "active_skip_rate": mean([_to_float(r, "skipped") for r in active]) if active else None,
        "active_valid_ratio_mean": mean([_to_float(r, "valid_ratio") for r in active]) if active else None,
        "active_grad_keep_ratio_mean": mean([_to_float(r, "grad_keep_ratio") for r in active]) if active else None,
        "latest_recommendation": target[-1].get("recommendation", "") if target else "",
    }
    if summary["reference_psnr_mean"] is not None and summary["active_psnr_mean"] is not None:
        summary["psnr_drop"] = summary["reference_psnr_mean"] - summary["active_psnr_mean"]
    else:
        summary["psnr_drop"] = None
    return summary


def suggest(summary):
    suggestions = []
    if summary["num_active_rows"] == 0:
        suggestions.append("No active propagation rows yet: lower --normal_prop_start or train farther into the propagation window.")
    if summary["active_skip_rate"] is not None and summary["active_skip_rate"] > 0.5:
        suggestions.append("Mask is skipped often: try --normal_prop_mask_mode rough_or_spec or lower --normal_prop_spec_thresh.")
    if summary["psnr_drop"] is not None and summary["psnr_drop"] > 0.5:
        suggestions.append("PSNR drops during propagation: try --normal_prop_normal_grad_scale 0.2 --normal_prop_ramp_iters 2000 --normal_prop_mask_mode spec_only.")
    if summary["active_grad_keep_ratio_mean"] is not None and summary["active_grad_keep_ratio_mean"] > 0.8:
        suggestions.append("Masked gradient is nearly unmasked: lower --normal_prop_max_valid_ratio or use --normal_prop_mask_mode rough_and_spec/spec_only.")
    if summary["active_grad_keep_ratio_mean"] is not None and 0 < summary["active_grad_keep_ratio_mean"] < 1e-4:
        suggestions.append("Propagation gradient is almost zero: inspect debug images and consider a lower spec threshold or a small grad-scale increase.")
    if not suggestions:
        suggestions.append("Diagnostics look stable; next sweep can vary one parameter at a time around the current setting.")
    return suggestions


def format_report(summary, suggestions):
    lines = ["# Normal Propagation Debug Report", ""]
    lines.append("## Metrics")
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Suggested next iterations")
    for item in suggestions:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default=None, help="Training output directory containing debug_normal_propagation/.")
    parser.add_argument("--debug_dir", default=None, help="Direct path to debug_normal_propagation/.")
    parser.add_argument("--write_report", action="store_true", help="Write report.md into the debug directory.")
    args = parser.parse_args()

    if args.debug_dir is None:
        if args.model_path is None:
            raise SystemExit("Pass --model_path or --debug_dir")
        args.debug_dir = os.path.join(args.model_path, "debug_normal_propagation")

    rows = load_rows(args.debug_dir)
    summary = summarize(rows)
    suggestions = suggest(summary)
    report = format_report(summary, suggestions)
    print(report)

    latest_path = os.path.join(args.debug_dir, "offline_summary.json")
    with open(latest_path, "w") as f:
        json.dump({"summary": summary, "suggestions": suggestions}, f, indent=2)
    if args.write_report:
        with open(os.path.join(args.debug_dir, "report.md"), "w") as f:
            f.write(report)


if __name__ == "__main__":
    main()
