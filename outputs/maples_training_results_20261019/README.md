# MAPLES training-results export

This directory contains a curated public export from the A7 execution records. It is not a full run archive; the A7 training implementation and full execution bindings are not published here.

- training_config_summary.csv: 17 actually trained runs / continuations; optimizer and learning-rate settings are normalized from the bound run configs. Alpha 1.0/2.0 and class 1/1, 2/2 baselines are reused formal runs, not duplicated training.
- loss_curve_200_update_windows.csv: mean and within-window SD of the recorded scalar loss for each non-overlapping 200-update window. Budget curves reuse their own formal seed-2026 steps 1-2000 and append the paired continuation steps 2001-4000.
- fixed_endpoint_per_image_metrics.csv: fixed-threshold endpoint metrics with per-image numerators, denominators, and NA reasons. Image keys are public MAPLES-DR image codes. Threshold remains 0.5.
- paired_comparison_summary.csv: frozen paired contrasts, image-level 95% intervals, valid image counts, and seed differences where applicable.
- paired_image_deltas.csv: paired per-image changes for the listed contrasts, sufficient to recompute macro means and inspect contributors.
- loss_curve.svg: summary plot of the numeric loss table.
- source_audit.json: sanitized protocol identifiers, SHA-256 identifiers, completion statuses, and export counts; no machine or private file paths.

No image pixels, labels, model weights, probability arrays, full training JSONL, pairing logs, raw configs, launch logs, host identifiers, or credentials are included. Official Test was not accessed. The Plan B AI visual annotations are not included; human visual review remains 0/105.

Recompute table invariants with:
    python scripts/verify_maples_training_export.py outputs/maples_training_results_20261019