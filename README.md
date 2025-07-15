# Belief State Geometry in Neural Networks

This repository implements experiments to demonstrate how neural networks represent belief state geometry in their internal activations, as described in the paper "Transformers Represent Belief State Geometry in their Residual Stream" ([arXiv:2405.15943](https://arxiv.org/pdf/2405.15943)).

## Overview

The project explores how LSTM networks trained on next-token prediction tasks develop internal representations that correspond to the geometry of belief states in hidden Markov models (HMMs). The key insight is that neural networks learn to represent the meta-dynamics of belief updating over hidden states of the data-generating process.

## Key Findings

- Neural networks trained on HMM-generated data develop internal representations that linearly encode belief state geometry
- The belief state geometry can exhibit complex fractal structure, which is accurately captured in the network's hidden states
- These representations contain information about the entire future, beyond local next-token prediction
- The geometry emerges over the course of training and is distributed across network layers

## Implementation

The notebook `activations_represent_belief_state_geometry.ipynb` contains:

- LSTM model implementation for sequence prediction
- Training on HMM-generated data with known belief state structure
- Analysis of hidden state activations using linear regression
- Visualization of belief state geometry in the probability simplex
- Comparison between random initialization and trained model representations

## Usage

1. Install dependencies: `pip install torch numpy matplotlib scikit-learn`
2. Run the Jupyter notebook to reproduce the experiments
3. The notebook saves plots showing the evolution of belief state geometry during training

## Theory

The work is grounded in computational mechanics and the theory of optimal prediction. The mixed-state presentation (MSP) formalism describes how optimal observers update their beliefs over hidden states given finite observations, leading to geometric structures that naturally correspond to neural network internal representations.

## Citation

If you use this code, please cite the original paper:

```
@article{shai2024transformers,
  title={Transformers Represent Belief State Geometry in their Residual Stream},
  author={Shai, Adam S. and Marzen, Sarah E. and Teixeira, Lucas and Oldenziel, Alexander Gietelink and Riechers, Paul M.},
  journal={arXiv preprint arXiv:2405.15943},
  year={2024}
}
```
