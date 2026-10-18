# EvidenceTrail

Implementation of **EvidenceTrail: Auditable Evidence Graphs for Process-Excursion Decision Support**.

## Installation

Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3, PyTorch 2.7.1+cu118, SciPy 1.15.3 and scikit-learn 1.6.1. Checks cover Python syntax, JSON/TOML parsing and CLI options against parser definitions; training and evaluation were not run.

## Data

The bundled two-hypothesis scripted session supplies fixed likelihoods, query costs, observations and explicit actor confirmations. It is generated session input; the actor is `scripted_operator`.

## Run

```bash
evidencetrail execute --input examples/scripted_session.json --output runs/full --variant full
```

## Ablations

Use the same input profile for each variant: `no_topology`, `no_voi`.

```bash
evidencetrail execute --input examples/scripted_session.json --output runs/no_topology --variant no_topology
```

## Outputs

`runs/full/session.json` stores final posterior, query costs and replay checks; `events.jsonl` is the appended evidence trail. Input and output paths are controlled by `--input` and `--output`.
Use a fresh output directory for a separate session because `events.jsonl` appends events.
