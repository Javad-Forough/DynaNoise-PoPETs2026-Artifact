# DynaNoise --- PoPETs 2026 Artifact (CIFAR-10)

This repository contains the official artifact for the accepted PoPETs
2026 paper:

**Dynamic Probabilistic Noise Injection for Membership Inference Defense**

This artifact reproduces the CIFAR-10 experimental results presented in
the paper and is designed for Artifact Evaluation.

We apply for the following badges:

-   Artifact Available\
-   Artifact Functional\
-   Artifact Reproduced

------------------------------------------------------------------------

# 1. Scope of the Artifact

This artifact reproduces **CIFAR-10 experiments only**.

The following components are included:

-   Target model (AlexNet) evaluation
-   HAMP defense evaluation
-   DynaNoise defense evaluation
-   All membership inference attacks:
    -   Confidence
    -   Loss
    -   Shadow
    -   LiRA
    -   Entropy (SM21)
    -   Modified Entropy (SM21)
-   MIDPUT computation
-   Deterministic data splits
-   Automatic CSV result generation
-   Runtime reporting

ImageNet-10 and SST-2 experiments reported in the paper are not included
in the artifact due to computational constraints during artifact
evaluation. The CIFAR-10 configuration provided here exercises the
complete experimental pipeline (target model evaluation, attacks,
defenses, and MIDPUT computation). The other datasets follow the same
pipeline implemented in the artifact.

------------------------------------------------------------------------

# 2. System Requirements

Tested on:

-   Python 3.10.12\
-   PyTorch 2.6.0 (CUDA 12.4 build)\
-   CUDA 12.x\
-   Ubuntu 22.04

GPU is strongly recommended.

Expected runtime (RTX 5000 or similar GPU):

-   HAMP: \~5--6 minutes\
-   DynaNoise: \~5--6 minutes\
-   Total runtime: \~11 minutes

CPU execution is supported but will be significantly slower.

------------------------------------------------------------------------

# 3. Installation

We recommend creating a clean Python environment.

``` bash
python -m venv pets_env
source pets_env/bin/activate
pip install --upgrade pip
```

## Step 1 --- Install PyTorch

Install PyTorch (CPU or GPU build) from https://pytorch.org.

Example for CUDA 12.4:

``` bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

If using CPU only:

``` bash
pip install torch torchvision
```

## Step 2 --- Install Remaining Dependencies

``` bash
pip install -r requirements.txt
```

Note: `requirements.txt` intentionally does not include PyTorch to avoid
CUDA version conflicts.

------------------------------------------------------------------------

# 4. Required Checkpoints

The artifact expects the following pretrained models:

Target models:

    saved_models/
        alexnet_cifar10_target.pt
        hamp_alexnet_cifar10.pt

Shadow models:

    saved_models_shadow/
        shadow_model.pt
        lira_shadow0.pt
        lira_shadow1.pt
        lira_shadow2.pt

If a required model is missing, execution stops with a clear error
message.


------------------------------------------------------------------------

# 5. Reproducing the Results

To reproduce the CIFAR-10 results supporting the main claims of the
paper:

    python reproduce_all.py

This script:

1.  Runs HAMP defense
2.  Runs DynaNoise defense
3.  Executes all membership inference attacks
4.  Computes MIDPUT
5.  Writes per-run CSV files
6.  Updates master CSV
7.  Prints runtime summary

The baseline **no-defense scenario** is automatically evaluated as part
of each run, allowing the artifact to report attack performance both
*before* and *after* applying each defense.

At completion:

    timing_summary.txt

will contain the runtime for each defense as well as the total runtime.

------------------------------------------------------------------------

# 6. Mapping to Paper Results

The command:

    python reproduce_all.py

reproduces the CIFAR-10 rows corresponding to:

-   The attack comparison table (HAMP vs DynaNoise)
-   The MIDPUT comparison table
-   The accuracy--privacy trade-off analysis

Terminal output and generated CSV files correspond directly to these
reported results.


------------------------------------------------------------------------

# 7. Output Files

Results are written to:

    results_artifact/

Each execution produces:

-   Unique per-run CSV
-   Updated master.csv
-   MIDPUT metrics
-   Attack accuracy (before/after defense)

------------------------------------------------------------------------

# 8. Determinism and Reproducibility

The artifact enforces:

-   Fixed random seeds
-   Deterministic dataset splits
-   Stable split hash
-   Pretrained checkpoints included


------------------------------------------------------------------------

# 9. Design for Reusability

The artifact is modular and structured for extension:

-   Separate model definitions
-   Separate defense implementations
-   Separate attack implementations
-   CSV-based output for further analysis

Researchers can easily extend this artifact to additional datasets or
architectures.

------------------------------------------------------------------------

# 10. Contact

For questions regarding artifact reproduction, please contact the
authors.

------------------------------------------------------------------------

End of README.
