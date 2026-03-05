# ARTIFACT-APPENDIX.md

## Paper Title

Dynamic Probabilistic Noise Injection for Membership Inference Defense

This artifact accompanies the accepted PoPETs 2026 paper listed above.

We apply for the following badges:

-   Artifact Available
-   Artifact Functional
-   Artifact Reproduced

------------------------------------------------------------------------

# 1. Artifact Available

## Public Access

The artifact is publicly available at a permanent repository link
provided in the HotCRP submission form.

The repository contains:

-   All source code required to reproduce the CIFAR-10 experiments
-   Pretrained model checkpoints
-   Reproduction scripts
-   Documentation (README)
-   requirements.txt
-   License file

The artifact is not behind a paywall and does not require manual
approval for access.

## License

The artifact is distributed under an open-source license included in the
repository.

## Relevance to the Paper

This artifact reproduces the CIFAR-10 experimental evaluation presented
in the paper, including:

-   Baseline model evaluation
-   HAMP defense evaluation
-   DynaNoise defense evaluation
-   All membership inference attacks
-   MIDPUT computation
-   Result logging to CSV files

------------------------------------------------------------------------

# 2. Artifact Functional

## Documentation

The README.md file provides:

-   System requirements
-   Installation instructions
-   Dependency setup
-   Required checkpoints
-   One-command reproduction instructions
-   Expected runtime
-   Output format description

All instructions have been tested in a clean environment.

## Completeness

The CIFAR-10 experimental pipeline described in the paper is fully
represented:

1.  Dataset loading (deterministic splits)
2.  Target model loading
3.  Shadow model loading
4.  LiRA shadow model loading
5.  Attack execution (Confidence, Loss, Shadow, LiRA, Entropy, Modified
    Entropy)
6.  Defense evaluation (HAMP, DynaNoise)
7.  MIDPUT computation
8.  CSV result generation

Other datasets reported in the paper (ImageNet-10 and SST-2) are not
included due to computational cost constraints, but follow the same
pipeline logic.

## Exercisability

The entire artifact can be executed with a single command:

    python reproduce_all.py


This script:

- Evaluates the baseline target model (no defense)
- Runs the HAMP defense
- Runs the DynaNoise defense
- Executes all membership inference attacks
- Computes the MIDPUT privacy–utility metric
- Writes results to CSV files
- Prints a runtime summary

No manual intervention is required.

Expected runtime on an RTX 5000 GPU:

-   HAMP: \~5--6 minutes
-   DynaNoise: \~5--6 minutes
-   Total runtime: \~11 minutes

CPU execution is supported but slower.

------------------------------------------------------------------------

# 3. Artifact Reproduced

## Core Claims Reproduced

This artifact reproduces the CIFAR-10 experimental results supporting
the following claims in the paper:

1.  DynaNoise significantly reduces membership inference attack
    accuracy.
2.  DynaNoise achieves higher MIDPUT compared to HAMP.
3.  DynaNoise preserves target model accuracy better than HAMP.
4.  All evaluated attacks are mitigated by DynaNoise.

## Mapping to Paper Results

The script:

    python reproduce_all.py

reproduces the CIFAR-10 rows of:

-   The attack comparison table (HAMP vs DynaNoise)
-   The MIDPUT comparison table
-   The accuracy vs privacy trade-off analysis

The output printed in the terminal and written to CSV files corresponds
directly to the reported CIFAR-10 results in the paper.

## Reproducibility Criteria

-   Deterministic random seed (seed=42)
-   Deterministic dataset split hash printed during execution
-   Pretrained checkpoints included
-   All attacks executed automatically
-   No retraining required

Exact numerical values may vary slightly due to differences in
hardware, CUDA versions, and PyTorch implementations. The artifact
uses deterministic dataset splits and fixed random seeds to minimize
randomness, but minor numerical variations across systems may still
occur. The qualitative conclusions reported in the paper remain
consistent (e.g., DynaNoise achieves higher MIDPUT than HAMP while
maintaining higher model accuracy).

------------------------------------------------------------------------

# 4. Limitations

-   ImageNet-10 and SST-2 experiments are not included due to resource
    constraints.
-   The artifact focuses on CIFAR-10 only.
-   GPU recommended for timely execution.

------------------------------------------------------------------------

# 5. Reusability

The artifact is structured to facilitate reuse:

-   Modular model loading
-   Modular defense implementation
-   Modular attack implementations
-   Clear CSV output for further analysis

Researchers can easily extend the artifact to additional datasets or
architectures.

------------------------------------------------------------------------

End of ARTIFACT-APPENDIX.md
