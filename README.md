# YOSO
You Only Shoot Once 
---
Single-frame Gerchberg-Saxton phase retrieval with AI-based data augmentation for in-line holography

## Features

- **Single-frame phase retrieval** reconstructs complex object information from a single digital in-line hologram.

- **Deep-learning-assisted multi-height hologram generation** uses supervised neural networks to numerically synthesize additional hologram at different defocus distance, enabling conventional dual-height phase retrieval from a single input hologram.

- **Physics-guided reconstruction pipeline** combines deep learning with the well-established Gerchberg–Saxton iterative phase retrieval algorithm, leveraging both data-driven prediction and physical wave propagation constraints.

- **Strong generalization capability** is achieved by training entirely on computer-generated holographic data derived from natural images, enabling application to diverse experimental samples without retraining.

- **Fast one-time training** enables efficient reconstruction after a single training procedure; the multi-scale ResNet model can be trained in less than two hours on a mid-range workstation.

- **Resolution-independent inference** enables direct processing of full-size holograms after training on smaller inputs, avoiding patch-and-stitch reconstruction.
---

## Project Structure

- `data_generation/` — generation of synthetic holographic training datasets using natural images and numerical wave propagation model  
- `training/` — training pipeline for the YOSO neural network  
- `reconstruction/` — hologram reconstruction pipeline, including second hologram estimation and Gerchberg–Saxton phase retrieval 

## Data Availability

The test dataset for running the code is available on Zenodo: [YOSO's Dataset](https://doi.org/10.5281/zenodo.19690495).

## How to Run

>**Note:** Before running the YOSO workflow, configure the required parameters in the YAML configuration files located in the `configs/` directories.

The YOSO workflow consists of three main steps:

1. Generation of synthetic training data
2. Training of the YOSO neural network
3. Reconstruction of in-line holograms

### 1. Generate training data

Generate synthetic holographic data used for training the YOSO model:

```bash
python data_generation/generate_dataset.py
```

### 2. Train model

Train the YOSO neural network using the generated datase:

```bash
python training/train.py
```

### 3. Reconstruct holograms

Use the trained YOSO model for numerical reconstruction of in-line holograms:

```bash
python reconstruction/reconstruct.py
```

## Installation

Clone the repository and install the required dependencies:
```bash
git clone https://github.com/AIMlab-WUT/YOSO.git
cd YOSO
pip install -r requirements.txt
```