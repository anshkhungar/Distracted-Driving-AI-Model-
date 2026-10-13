# Distracted Driving Detection | Computer Vision

A computer vision exploration of **driver distraction recognition**, using machine learning to study visual cues associated with potentially unsafe driving behavior.

**Focus:** Computer Vision · Machine Learning · Road Safety · Python

## Overview

Distracted driving remains a major road-safety problem. This project investigates how image-based machine learning can support detection and analysis of driver behavior. The original experimentation was developed in Google Colab as part of an NSLC project.

The repository is organized to distinguish the **exploratory model work** from **reusable evaluation utilities**. The original notebook is preserved rather than presented as a production-ready application.

## Repository structure

```text
.
├── notebooks/
│   └── distracted_driving_exploration.ipynb  # Original Colab research
├── src/
│   └── distracted_driving/
│       ├── __init__.py
│       └── metrics.py                         # Reusable classification metrics
├── scripts/
│   └── evaluate_predictions.py                # Evaluate exported predictions
├── tests/
│   └── test_metrics.py
├── pyproject.toml
└── README.md
```

## What the project explores

- Using computer vision and machine-learning workflows to examine distracted driving
- Experimentation and analysis in a Python notebook
- Evaluating image-classification outputs with accuracy and per-class precision, recall, and F1 scores

> **Project status:** Research/prototype. The original model training and experiments are in the [exploration notebook](notebooks/distracted_driving_exploration.ipynb). The Python utilities here evaluate prediction files; they are **not** a standalone pretrained driver-monitoring system or a real-time inference app.

## Evaluate prediction results

The evaluation utility accepts a CSV with **`y_true`** and **`y_pred`** columns containing the actual and predicted class labels.

```bash
python -m pip install -e ".[dev]"
python scripts/evaluate_predictions.py path/to/predictions.csv
python -m pytest
```

Example CSV:

```csv
y_true,y_pred
attentive,attentive
distracted,attentive
distracted,distracted
```

It prints accuracy and a classification report. This is designed for **exported classification results**; it does not train a new model.

## Research notebook

Open [the original Colab notebook](notebooks/distracted_driving_exploration.ipynb) for the underlying experimental work. Training data, pretrained weights, and datasets are not included in this repository; reproducibility of the original notebook may depend on its external setup.

## Future development

Potential next steps include extracting the original training pipeline into standalone modules, documenting the dataset and label definitions, exporting verified model weights, and adding a reproducible inference entry point. These are planned improvements, not implemented features.

## Responsible use

Visual classification can be affected by camera placement, lighting, occlusion, and dataset bias. This prototype should **not** be used to make safety-critical driving decisions.
