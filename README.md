Advanced Behavioral Analysis Pipeline using GRU Networks

This project implements a robust, memory-optimized machine learning pipeline designed to analyze complex animal behavior from video tracking data. It leverages Gated Recurrent Units (GRU) to detect and classify specific behavioral patterns, with a specialized focus on identifying rare and challenging events (such as aggression) in imbalanced datasets.

📋 Table of Contents

Key Features

Project Architecture

Installation

Usage

Imbalance Strategies

Evaluation

Contributing

License

🚀 Key Features

Memory-Optimized Pipeline: Built to handle large-scale datasets (1000+ videos) using disk-based batch processing, downcasting, and aggressive garbage collection to prevent RAM overflow.

Advanced Feature Engineering: Automatically extracts kinematic and spatial metrics, including:

Body/Face orientation and angular velocities.

Centroid speeds and accelerations (pixel & cm units).

Relative distances (nose-to-forepaw, inter-forepaw).

Spatial context (distance to walls, distance to center).

Robust Imbalance Handling: Implements multiple strategies to tackle class imbalance:

Smart Balancing: Downsamples majority classes while oversampling minority classes.

Hybrid Strategy: Combines synthetic oversampling (with noise jitter) and calculated class weights.

Behavioral Inference Engine: A post-processing logic layer that addresses the "data ceiling" problem for rare behaviors. It allows the detection of low-confidence events by validating them against a temporal context history (e.g., detecting an attack because it was preceded by a chase).

📂 Project Architecture

1. Data Ingestion & Processing (DataModeling)

The DataModeling class orchestrates the loading and cleaning of raw tracking data.

Interpolation: Handles missing frames and smoothes tracking jitter.

Feature Extraction: Vectorized calculation of 29+ geometric and kinematic features.

Windowing: Slices continuous data into fixed-size temporal windows (default: 105 frames) suitable for sequence modeling.

2. Large-Scale Batch Processing

To handle datasets larger than system RAM:

main_optimized(): The entry point for large datasets. It splits videos into batches, processes them, saves intermediate parquet files to disk, and aggregates them only when necessary.

process_video_batch(): Processes chunks of videos in parallel or sequentially, ensuring memory is cleared between batches.

3. The Model (BehaviorGRU)

A specialized Recurrent Neural Network:

Input: Sequence of features (Window Size × Feature Count).

Architecture: Multi-layer GRU followed by a fully connected classification head.

Initialization: Xavier Uniform initialization for stable convergence.

4. Behavioral Inference Engine

A "Magic Fix" for rare class detection (e.g., Attack, Chase) where training samples are scarce.

Low Thresholding: Accepts predictions with low probability (e.g., >10%) if specific conditions are met.

Context Awareness: Maintains a history buffer (e.g., last 10 frames) to check for precursor behaviors (e.g., "Chase" often precedes "Attack").

⚙️ Installation

Clone the repository:

git clone [https://github.com/yourusername/behavioral-analysis-pipeline.git](https://github.com/yourusername/behavioral-analysis-pipeline.git)
cd behavioral-analysis-pipeline


Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`


Install dependencies:

pip install pandas numpy torch scikit-learn psutil


💻 Usage

Training the Model

Use the train_with_strategy function to run the full training loop with your desired configuration.

from pipeline import train_with_strategy

# Example: Train using the 'hybrid' strategy to fix class imbalance
model, label_encoder, scaler, class_weights, results = train_with_strategy(
    balance_strategy='hybrid',
    epochs=50,
    resample_config={'target_min_samples': 1000},
    apply_attack_rules=True,
    attack_rules_config={
        'low_threshold': 0.3, 
        'context_window': 3
    }
)


Processing Large Datasets

For processing 100+ videos, use the optimized main function to avoid OOM errors.

from pipeline import main_optimized

X_tensor, y_tensor, label_encoder, scaler, _, _ = main_optimized(
    balance_strategy='weights',
    batch_size_videos=50,  # Process 50 videos at a time
    window=105
)


⚖️ Imbalance Strategies Explained

Strategy

Description

Best For

Weights

Uses weighted CrossEntropyLoss. No data modification.

Large datasets, production speed.

Resample

Duplicates minority samples (with noise) to match a target count.

Very rare classes.

Hybrid

Moderate resampling + Class weights.

Best overall performance.

Smart

Downsamples majority (background) + Oversamples minority.

massive datasets dominated by background.

📊 Evaluation Strategies

The pipeline provides comprehensive evaluation metrics:

Standard Evaluation: Accuracy, Precision, Recall, and F1-Score per class.

Logic-Enhanced Evaluation: Compares raw model performance vs. performance with the BehavioralInferenceEngine applied.

Confusion Matrix: Visualizes misclassifications.

🤝 Contributing

Contributions are welcome! Please follow these steps:

Fork the project.

Create your feature branch (git checkout -b feature/AmazingFeature).

Commit your changes (git commit -m 'Add some AmazingFeature').

Push to the branch (git push origin feature/AmazingFeature).

Open a Pull Request.

📝 License

This project is open-source. Please attribute the original authors when using this pipeline for research or production.
