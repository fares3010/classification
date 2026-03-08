# Memory-Optimized Mouse Behavior Detection

A memory-efficient pipeline for the **MABe (Mouse Action/Behavior) Challenge**, which detects social actions between mice from pose-tracking data. This is a refactored version of the original v7 pipeline, redesigned to run comfortably within **~8 GB RAM** on Kaggle CPU kernels.

---

## Table of Contents

1. [Overview](#overview)  
2. [Architecture](#architecture)  
3. [Memory Optimizations](#memory-optimizations)  
4. [Requirements](#requirements)  
5. [Data Layout](#data-layout)  
6. [Configuration](#configuration)  
7. [Pipeline Stages](#pipeline-stages)  
8. [Output](#output)  
9. [Troubleshooting](#troubleshooting)

---

## Overview

The pipeline classifies frame-level social actions (e.g., sniffing, chasing, mounting) between pairs of mice using:

- **Pose-tracking parquet files** containing per-frame body-part coordinates for each mouse.
- **Annotation parquet files** with labeled action segments (agent, target, action, start/stop frame).
- **Metadata** from `train.csv` / `test.csv` (fps, arena dimensions, sex, strain, etc.).

It produces a `submission.csv` with predicted action segments for test videos.

---

## Architecture

The pipeline follows a **2-pass** design so that each video's tracking data is loaded at most once per pass:

```
Pass 1 ──► Reservoir-sampled descriptors ──► PCA + KMeans clustering
Pass 2 ──► Full features + cluster features ──► Flush chunks to disk (.npz)
Train  ──► Stream chunks from disk ──► One XGBoost model per action
Infer  ──► Load test videos one at a time ──► Predict + post-process
```

### Feature Groups

Each agent–target pair at each frame gets features from four sources:

| Group | Prefix | Description |
|-------|--------|-------------|
| Single-mouse | `A_`, `B_` | Speed, acceleration, body length, heading, ear distance |
| Pairwise | (none) | Inter-mouse distance, approach rate, facing angle, nose-nose distance |
| Third-party | `t` | Distance to nearest other mouse, crowding metrics |
| Cluster | `cA_`, `cB_`, `cP_` | KMeans cluster ID, distance to centroids, cluster histograms |
| Metadata | `as`, `ts`, `fp`, etc. | Sex, strain, arena shape, fps, self-action flag |

---

## Memory Optimizations

This refactored version addresses several memory bottlenecks present in the original v7 code:

### 1. Reservoir Sampling (Pass 1)

The original code accumulated all snippet arrays into unbounded Python lists before stacking them into a single NumPy array. With 2000 videos this could exceed several GB.

**Fix:** A `Reservoir` class pre-allocates a fixed-size buffer (`MAX_RESERVOIR_S = 150K` rows for single, `MAX_RESERVOIR_P = 150K` for pair descriptors). Incoming rows are accepted via reservoir sampling, so memory stays constant regardless of how many videos are processed. Video count and per-video sample size are also reduced (`UNSUP_MAX_VIDS`: 2000→1200, `SNIPPET_SAMPLE`: 120→80).

### 2. Disk-Backed Feature Chunks (Pass 2)

The original code kept every video's feature DataFrame in `data[act]["X"]` lists in RAM, then concatenated them all at training time. This created a peak memory spike equal to roughly 2× the total feature data.

**Fix:** Each video's sampled features and labels are immediately flushed to a compressed `.npz` file in a temporary directory. Only a lightweight registry of file paths and row counts stays in memory. Total in-memory footprint during Pass 2 is proportional to a single video, not the full dataset.

### 3. Capped Streaming at Training Time

The original code concatenated all chunks into a single array before downsampling negatives, doubling peak memory at training time.

**Fix:** `_load_and_cap_chunks()` loads `.npz` files one at a time with a hard cap (`MAX_ROWS_PER_ACT = 300K`). If total rows exceed the cap, negatives are subsampled per-chunk during loading, so the full dataset never materializes in memory at once.

### 4. Aggressive Cleanup

- `del df; gc.collect()` after pivot operations in `load_track()`
- `del raw` immediately after adding to the reservoir
- `del pvid, sc; gc.collect()` after processing each video
- Temporary `.npz` directory cleaned up after training completes

### Memory Profile Comparison (approximate)

| Stage | Original v7 Peak | v7-LOWMEM Peak |
|-------|-------------------|----------------|
| Pass 1 snippets | ~2–4 GB | ~0.5 GB (capped) |
| Pass 2 features | ~4–8 GB | ~0.5 GB (single video) |
| Training concat | ~4–6 GB | ~1–2 GB (streamed) |
| **Overall peak** | **~8–12 GB** | **~2–4 GB** |

---

## Requirements

```
python >= 3.8
numpy
pandas
scipy
scikit-learn
xgboost
joblib
```

## Data Layout

The pipeline expects the following directory structure:

```
<root>/
├── train.csv
├── test.csv
├── train_tracking/
│   └── <lab_id>/
│       └── <video_id>.parquet
├── train_annotation/
│   └── <lab_id>/
│       └── <video_id>.parquet
└── test_tracking/
    └── <lab_id>/
        └── <video_id>.parquet
```

The root directory is auto-detected. It checks (in order): `/kaggle/input/*/`, several hard-coded Kaggle/Colab paths, `./data`, and `.`.

### Tracking Parquets

Each file contains columns: `video_frame`, `mouse_id`, `bodypart`, `x`, `y`. Expected body parts include `nose`, `neck`, `body_center`, `ear_left`, `ear_right`, `tail_base`, among others.

### Annotation Parquets

Each file contains columns: `agent_id`, `target_id`, `action`, `start_frame`, `stop_frame`.

### CSV Metadata

`train.csv` and `test.csv` include per-video metadata: `video_id`, `lab_id`, `frames_per_second`, `pix_per_cm_approx`, `behaviors_labeled`, `arena_width_cm`, `arena_height_cm`, `arena_shape`, `mouse1_sex`, `mouse1_strain`, etc.

---

## Configuration

Key constants at the top of the script:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `UNSUP_MAX_VIDS` | 1200 | Max videos sampled for clustering |
| `SNIPPET_SAMPLE` | 80 | Frames per mouse/pair per video for clustering |
| `MAX_RESERVOIR_S` | 150,000 | Max rows in single-mouse descriptor reservoir |
| `MAX_RESERVOIR_P` | 150,000 | Max rows in pair descriptor reservoir |
| `N_CLUST_S` | 50 | Single-mouse KMeans clusters |
| `N_CLUST_P` | 80 | Pair KMeans clusters |
| `MAX_ROWS_PER_ACT` | 300,000 | Hard cap on training rows per action |
| `NEG_RATIO` | 5.0 | Negative-to-positive sampling ratio |
| `PAD_FRAMES` | 150 | Context frames around positive labels |
| `MAX_FRAMES_VID` | 50,000 | Per-video sample cap |

To reduce memory further, lower `MAX_RESERVOIR_S`, `MAX_RESERVOIR_P`, and `MAX_ROWS_PER_ACT`.

---

## Pipeline Stages

### Pass 1 — Snippet Collection

Iterates over up to `UNSUP_MAX_VIDS` training videos. For each mouse and each mouse pair, extracts compact descriptors (speed at multiple lags, body length changes, heading angular velocity, ear distance, inter-mouse distance, velocity alignment, facing angle, nose-nose distance). Descriptors are streamed into fixed-size reservoir buffers.

### Clustering

Fits `StandardScaler` → `PCA` → `MiniBatchKMeans` on the reservoir contents. Produces two cluster models: one for single-mouse behavior states, one for pairwise interaction states.

### Pass 2 — Labeled Feature Extraction

Iterates over all labeled training videos. For each annotated agent–target–action triple:

1. Builds the full feature vector (single + pair + third-party + cluster + metadata).
2. Samples frames around positive labels plus a controlled number of negatives.
3. Flushes the result as a compressed `.npz` to a temp directory.

### Training

For each action, loads chunks from disk (with a hard row cap), trains an `XGBClassifier` with early stopping on a video-level train/val split, and tunes the decision threshold by sweeping for best F1.

### Inference

Loads each test video once, builds features for every behavior triple listed in `test.csv`, predicts frame-level probabilities, applies smoothing + thresholding + gap-merging to produce action segments.

---

## Output

### submission.csv

| Column | Type | Description |
|--------|------|-------------|
| `row_id` | int | Sequential row index |
| `video_id` | int | Video identifier |
| `agent_id` | str | Mouse performing the action |
| `target_id` | str | Target mouse (or "self") |
| `action` | str | Action label |
| `start_frame` | int | Segment start frame (inclusive) |
| `stop_frame` | int | Segment end frame (inclusive) |

### Saved Models

Per-action XGBoost models are saved to `<output_dir>/models/m_<action>.joblib`, each containing `(model, feature_columns, threshold)`.

---

## Troubleshooting

**Out of memory during Pass 1:** Lower `UNSUP_MAX_VIDS` and `MAX_RESERVOIR_S` / `MAX_RESERVOIR_P`.

**Out of memory during training:** Lower `MAX_ROWS_PER_ACT`. The pipeline already prioritizes keeping positive samples and subsampling negatives.

**Empty submission:** Check that test tracking parquets exist and that `behaviors_labeled` in `test.csv` lists actions that were also present in training data.

**Slow performance:** The bottleneck is typically I/O (loading parquets). Ensure data is on SSD. Reducing `UNSUP_MAX_VIDS` helps Pass 1 speed; the training phase is usually fast since XGBoost uses `tree_method="hist"`.

**Column mismatch warnings:** If test videos have different body parts than training videos, missing feature columns are filled with zeros. This is handled automatically by `reindex(columns=fcols, fill_value=0.0)`.
