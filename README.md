# Advanced Behavioral Analysis using GRU Networks

This project implements a comprehensive, memory-optimized pipeline for analyzing complex animal behavior (specifically murine/mouse models) from video tracking data. It leverages **Gated Recurrent Units (GRU)** and a sophisticated **Behavioral Inference Engine** to detect specific behavioral patterns, with a specialized focus on identifying rare and challenging events like aggression.

## 🚀 Key Features

* **Memory-Optimized Pipeline:** Designed to handle large-scale datasets (1000+ videos) on systems with limited RAM (16GB-32GB) using disk-based batch processing, type downcasting, and generator-based iteration.
* **Robust Feature Engineering:** Vectorized extraction of kinematic features (velocity, acceleration, angular speed) and spatial geometries (body-part distances, orientations, wall distances).
* **Advanced Imbalance Handling:** Implements 8+ strategies to handle class imbalance, including "Smart Balancing" (downsampling majority + oversampling minority with noise) and Hybrid approaches.
* **Behavioral Inference Engine:** A post-processing logic layer that solves the "data ceiling" problem for rare behaviors by using low-probability thresholds validated by temporal context rules.
* **Scalable Architecture:** Capable of processing disparate video metadata, smoothing noisy tracking data, and dynamically windowing time-series data.

## 🛠️ Pipeline Overview

1.  **Data Ingestion:** Loads video metadata, annotations, and raw body-part coordinates (Parquet/CSV).
2.  **Preprocessing:**
    * Optimizes DataFrame memory footprint.
    * Interpolates missing tracking data.
    * Downsamples video framerates for efficiency.
3.  **Feature Engineering:** Transforms raw coordinates into meaningful features (e.g., `nose_forepaw_distance`, `body_orientation`, `face_velocity`).
4.  **Windowing:** Slices continuous data into fixed-size temporal windows with specific handling for "Action" vs "Background" segments.
5.  **Model Training:** Trains a GRU-based neural network using PyTorch with Xavier initialization and Stratified Splitting.
6.  **Inference:** Applies the Behavioral Inference Engine to refine predictions.

## 📦 Installation

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install pandas numpy torch scikit-learn psutil openai
