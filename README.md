# NVIDIA Isaac MLOps PoC

Automated MLOps pipeline for NVIDIA's synthetic data generation and model training workflow, running on OpenShift AI (RHOAI 3.3).

Takes the manual, multi-hour process from NVIDIA's Isaac Sim Module 3 — generate synthetic images, convert annotations, train an object detection model, evaluate results — and wraps it in a single parameterized pipeline with experiment tracking and model registration.

## What It Does

```
Isaac Sim (generate synthetic palletjack images)
  → TAO Toolkit (convert to TFRecords, train DetectNet_v2)
    → MLflow (log metrics, register model if mAP > threshold)
```

All orchestrated by Data Science Pipelines (KFP v2) on OpenShift with GPU scheduling via the NVIDIA GPU Operator.

## Repository Structure

```
charts/isaac-mlops-poc/    Helm chart — deploys all infrastructure
  templates/                 MinIO, DSPA, MLflow, PVCs, RBAC, TAO specs, test Jobs
  values.yaml                All tunables (GPU type, frame count, epochs, etc.)

pipelines/
  isaac_training_pipeline.py   5-step KFP v2 pipeline definition (compile + upload)
  test_pipeline.py             Smoke test to verify DSPA is functional

planning/                  PoC plan, phase results, architecture decisions
notes/                     Detailed walkthrough of how everything works
```

## Prerequisites

- OpenShift 4.x with RHOAI 3.3+ (DSPA, MLflow operator, Kueue)
- NVIDIA GPU Operator with an L40S or equivalent GPU node
- NGC API key ([get one here](https://org.ngc.nvidia.com/setup/api-key))
- TAO Toolkit EULA accepted on [NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/containers/tao-toolkit)

## Quick Start

**1. Deploy infrastructure:**

```bash
helm install isaac-mlops charts/isaac-mlops-poc/ \
  --set ngc.apiKey=<your-key> \
  -n isaac-mlops-poc --create-namespace
```

**2. Post-install (link pull secret to pipeline SA):**

```bash
oc secrets link pipeline-runner-isaac-pipelines ngc-secret \
  --for=pull -n isaac-mlops-poc
```

**3. Compile and submit the pipeline:**

```bash
pip install -r requirements.txt
python pipelines/isaac_training_pipeline.py
# Upload pipelines/isaac_training_pipeline.yaml via the RHOAI dashboard
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_frames` | 100 | Synthetic images to generate (5000 for production) |
| `epochs` | 2 | Training epochs (80-100 for production) |
| `batch_size` | 4 | Images per GPU batch |
| `map_threshold` | 0.0 | Minimum mAP to register the model |
| `gpu.product` | NVIDIA-L40S | GPU node selector label |

## NVIDIA Container Images

| Image | Purpose | Size |
|-------|---------|------|
| `nvcr.io/nvidia/isaac-sim:4.5.0` | Synthetic data generation (Replicator) | ~15GB |
| `nvcr.io/nvidia/tao/tao-toolkit:5.0.0-tf1.15.5` | TFRecord conversion + DetectNet_v2 training | ~8GB |

## Documentation

See [`notes/mlops-poc-walkthrough.md`](notes/mlops-poc-walkthrough.md) for a detailed explanation of every component, how data flows between pipeline steps, and what all the ML concepts mean.
