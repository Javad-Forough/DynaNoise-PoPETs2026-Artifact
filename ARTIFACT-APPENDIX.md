# Artifact Appendix

Paper title: **Dynamic Probabilistic Noise Injection for Membership
Inference Defense**

Requested Badge(s): 
- \[x\] **Available** 
- \[x\] **Functional** 
- \[x\] **Reproduced**

------------------------------------------------------------------------

## Description

This artifact accompanies the accepted PoPETs 2026 paper:

**Dynamic Probabilistic Noise Injection for Membership Inference
Defense**

The artifact provides all code, scripts, and pretrained models necessary
to reproduce the CIFAR-10 experimental evaluation presented in the
paper.

Specifically, the artifact enables reproduction of:

- Baseline model evaluation (no defense)
- HAMP defense evaluation
- DynaNoise defense evaluation
- Membership inference attacks (Confidence, Loss, Shadow, LiRA, Entropy,
  Modified Entropy)
- MIDPUT privacy–utility metric computation
- Result logging and runtime reporting

All experiments are automated through a single reproduction script.

### Security/Privacy Issues and Ethical Concerns

The artifact does not execute malicious code and does not modify system
security settings. It trains and evaluates machine learning models using
the publicly available CIFAR-10 dataset.

The artifact downloads CIFAR-10 automatically via
`torchvision.datasets.CIFAR10`.

No personal data or sensitive datasets are used.

------------------------------------------------------------------------

## Basic Requirements

### Hardware Requirements

Minimal hardware requirements:

- Standard CPU machine capable of running Python
- At least **8 GB RAM**

GPU is **not required**, but strongly recommended for faster execution.

Recommended hardware (used during development):

- NVIDIA RTX 5000 GPU
- 16 GB RAM

### Software Requirements

The artifact was tested on:

- Ubuntu 22.04
- Python 3.10
- PyTorch 2.6.0
- CUDA 12.x

Required Python dependencies are listed in:

`requirements.txt`

Main dependencies include:

- torch
- torchvision
- numpy
- scikit-learn
- tqdm

### Machine Learning Models

The artifact includes pretrained models required to reproduce the
results:

`saved_models/` `saved_models_shadow/`

These include:

- Target model (AlexNet trained on CIFAR-10)
- HAMP defense model
- Shadow models for attack training
- LiRA shadow models

### Datasets

The artifact uses the **CIFAR-10 dataset**, which is automatically
downloaded during execution using:

`torchvision.datasets.CIFAR10(download=True)`

No manual dataset preparation is required.

### Estimated Time and Storage Consumption

Approximate runtime (RTX 5000 GPU):

- HAMP evaluation: ~5–6 minutes
- DynaNoise evaluation: ~5–6 minutes
- Total runtime: ~11 minutes

CPU execution is possible but slower.

Disk space requirements:

- Artifact repository: ~1.5 GB (including pretrained models)
- CIFAR-10 dataset download: ~170 MB

------------------------------------------------------------------------

## Environment

### Accessibility

The artifact is publicly available at the following repository:

https://github.com/Javad-Forough/DynaNoise-PoPETs2026-Artifact

The repository includes:

- Source code
- Pretrained checkpoints
- Reproduction scripts
- Documentation
- License

The artifact is distributed under the **MIT License**.

### Set up the environment

Clone the repository:

``` bash
git clone https://github.com/Javad-Forough/DynaNoise-PoPETs2026-Artifact.git
cd DynaNoise-PoPETs2026-Artifact
```

Create a Python environment:

``` bash
python -m venv pets_env
source pets_env/bin/activate
pip install --upgrade pip
```

Install PyTorch (example CUDA version):

``` bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Install remaining dependencies:

``` bash
pip install -r requirements.txt
```

### Testing the Environment

To verify that the artifact runs correctly, execute:

``` bash
python reproduce_all.py
```

Successful execution will:

- Run baseline evaluation
- Run HAMP defense
- Run DynaNoise defense
- Execute all membership inference attacks
- Compute MIDPUT metrics
- Produce CSV result files
- Print runtime summary

The script will also generate:

`timing_summary.txt`

------------------------------------------------------------------------

## Artifact Evaluation

### Main Results and Claims

The artifact reproduces the CIFAR-10 experimental results reported in the paper and validates the following claims:

**Claim 1 (Privacy Improvement).**  
DynaNoise reduces the Attack Success Rate (ASR) of multiple membership inference attacks compared to the undefended baseline, including Confidence, Loss, Entropy, Modified Entropy (M-Entropy), Shadow-model, and LiRA attacks.

**Claim 2 (Privacy–Utility Trade-off).**  
DynaNoise achieves higher MIDPUT scores than the HAMP defense, indicating a stronger balance between privacy protection and model accuracy.

**Claim 3 (Utility Preservation).**  
DynaNoise maintains target model accuracy close to the baseline model while providing privacy protection.

**Claim 4 (Robustness Across Attacks).**  
The defense consistently reduces attack effectiveness across different classes of membership inference attacks, including metric-based, shadow-model, and likelihood-ratio attacks.

### Experiments

#### Experiment 1: CIFAR-10 Defense Evaluation

Time:

- ~11 compute minutes on RTX 5000 GPU

Execution:

``` bash
python reproduce_all.py
```

This script:

- Evaluates the baseline model (no defense)
- Runs HAMP defense
- Runs DynaNoise defense
- Executes all membership inference attacks
- Computes MIDPUT metrics
- Writes results to CSV files

The output printed in the terminal and stored in CSV files corresponds
to the CIFAR-10 experimental results reported in the paper.

------------------------------------------------------------------------

## Limitations

The artifact reproduces **CIFAR-10 experiments only**.

ImageNet-10 and SST-2 experiments reported in the paper are not included
due to computational constraints during artifact evaluation.

These datasets follow the same experimental pipeline implemented in the
artifact.

GPU execution is recommended for timely experiment completion.

------------------------------------------------------------------------

## Notes on Reusability

The artifact is structured modularly to support extension and reuse.

Researchers can easily adapt the artifact to:

- Additional datasets
- Different model architectures
- Alternative defense mechanisms
- New membership inference attacks

The modular design of the codebase facilitates experimentation with
privacy defenses in machine learning systems.
