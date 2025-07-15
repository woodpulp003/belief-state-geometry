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

Checkpoint models already saved till 100k epoch training. If you wish to get more epochs/change architecture; run the entire notebook and change the hyperparameters.
