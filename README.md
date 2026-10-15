# EvidenceTrail

Implementation of **EvidenceTrail: Auditable Evidence Graphs for Process-Excursion Decision Support**.

A calibrated diagnostic backend is separate from the evidence-session engine. Accepted signed likelihood-ratio contributions update the posterior; topology, availability, and query history restrict feasible inspections. Expected entropy reduction is penalized by cost, latency, and acquisition risk. Explicit actor edits, counterevidence status, confidence override, and disposition confirmation are preserved in an append-only JSONL trail with pre/post states. Scripted session replay is not a substitute for a human study.

## Install

Use Python 3.10 or later. Create and activate a virtual environment, then install:

```bash
python -m pip install -e .
```

## Prepare data

Backend training uses `x[N,D]` evidence descriptors and cause `y[N]`. Execute consumes initial backend logits, hypothesis-stage neighborhoods, query likelihood matrices (outcome by cause), costs, available outcomes, and a chronological list of actor commands. Commands explicitly accept/reject evidence, state a rationale, and optionally confirm disposition or escalation. Never insert unrevealed outcomes into backend features.

Arrays use numeric NPZ format, loaded without pickle. `group[N]` identifies independent subjects, devices, scenes, lots, or source images. Related samples must share a group. Optional `split[N]` codes 0/1/2/3 explicitly assign training/validation/calibration/test (three-role input using 0/1/2 remains supported); otherwise ordered groups are allocated 60/15/10/15 to training/validation/calibration/test. Test data never select model parameters or checkpoints.

```bash
evidencetrail demo --output data/example.npz --count 96 --config config.json
evidencetrail prepare --input data/example.npz --output data/prepared
```

The demo defines a small synthetic problem with the documented tensor interface. Use application data in that schema for the paper's evaluation settings.

## Train and execute

```bash
evidencetrail train --input data/prepared --output runs/full --config config.json --variant full --epochs 100 --batch-size 16
evidencetrail evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
evidencetrail plot --input runs --output runs/figures
```

Training uses AdamW and selects the minimum validation objective. `model.pt` stores parameters and configuration; `history.json` records training/validation objectives. Evaluation exports predictions and mechanism tensors to `predictions.npz` and computed scores to `metrics.json`. Plotting exports comparison charts from those actual evaluation files.

`--device cpu` is the default; `--device cuda` selects CUDA. `--lr`, `--epochs`, `--batch-size`, and `--seed` control optimization. Method parameters are in `config.json`; `--set key=value` overrides one value using JSON syntax. Use identical splits and seeds across comparisons.

## Calibration

For interval or classification confidence calibration, use independent groups excluded from fitting and checkpoint selection:

```bash
evidencetrail calibrate --input data/prepared/calibration.npz --model runs/full/model.pt --output runs/full/calibrated.pt
```

Evaluate `calibrated.pt` with the same evaluation command. Classification uses temperature scaling; numeric intervals use a 90% group-maximum split-calibration expansion. Insufficient independent units retain infinite intervals. Pretraining checkpoints can initialize supervised training with `train --init path/to/model.pt`.

## Baselines and ablations

Available variants: `full`, `baseline`, `no_topology`, `no_voi`. `baseline` removes the paper-specific mechanism while retaining the input and prediction task. Other variants disable their named component.

```bash
evidencetrail train --input data/prepared --output runs/baseline --config config.json --variant baseline
evidencetrail evaluate --input data/prepared/test.npz --model runs/baseline/model.pt --output runs/baseline/evaluation
```

## Metrics and implementation conventions

Classification reports accuracy, balanced accuracy, macro-F1, NLL, confidence ECE, and worst-group F1; binary tasks add AUROC and AUPRC. Unknown labels use -1 and enable unknown AUROC and OSCR. Segmentation reports Dice, IoU, precision, recall, MAE, Brier/ECE, two-pixel boundary F1, symmetric contour distance, and HD95. Reconstruction reports RMSE, MAE, signed bias, unit-range PSNR, and Gaussian-window SSIM. Each method adds its mechanism metrics where applicable.

Parameter counts and forward timing describe the executing process. Use batch size one and a declared device configuration for device latency comparisons. Physical energy and scores from external task evaluators require measured inputs; they are not inferred from parameter counts. All prediction arrays remain available for task-specific evaluation.

The implementation uses the explicit core operations described above with configurable compact networks. Numeric conventions and application-specific information are stated with the method and data schema. The parameter file contains the current defaults.

## Execute the mechanism

```bash
evidencetrail execute --input example.json --output runs/execution --variant full
```

The example is an explicit analytical/simulated input specification, not an experiment result. Replace it with the schema described above for application data. The execute path writes the actual selected actions, constraints, and mechanism metrics.
