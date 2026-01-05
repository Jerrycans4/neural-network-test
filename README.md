This project implements a Multi-Layer Perceptron (MLP) from first principles. Instead of using high-level libraries, the optimization is handled via manual derivation of gradients.

Numerical Considerations:
Normalization: Input features are scaled to [0,1] to prevent exponential overflow in the Softmax function.
Weight Initialization: Used He Initialization (W∼N(0,2/n​)) to maintain variance of activations across layers and avoid vanishing gradients.
