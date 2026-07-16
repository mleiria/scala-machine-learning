agy --conversation=67feab92-ff6c-45a5-ac36-7a5fcf322dc8



# Scala Machine Learning Project

This project implements various machine learning algorithms from scratch using the `breeze` linear algebra library in Scala.

## Features

### 1. Linear and Logistic Regression
- Core implementations of Linear Regression (MSE) and Logistic Regression (Binary Cross-Entropy).
- Generic Gradient Descent optimizer used across models.

### 2. Long Short-Term Memory (LSTM)
A complete implementation of an LSTM network for sequential data processing.

#### Core Architecture
The LSTM is implemented in the `pt.mleiria.core.nn` package, separating weights, internal state, and optimization logic.

#### Key Components
- **`LSTMWeights`**: Holds weight matrices ($W_f, W_i, W_c, W_o$) and bias vectors ($b_f, b_i, b_c, b_o$) with Xavier (Glorot) initialization.
- **`LSTMState`**: Stores intermediate values for each time step required for Backpropagation Through Time (BPTT).
- **`LSTM` object**:
    - `forward(...)`: Implements the LSTM cell logic over a sequence.
    - `backward(...)`: Implements BPTT to compute gradients for all weights and biases.
    - `computeCost(...)`: Calculates the Mean Squared Error (MSE) for sequences.
- **`LSTMOptimizer`**: Handles the training loop and parameter updates using Stochastic Gradient Descent (SGD).

#### Mathematical Logic
- **Forward Pass**:
    - $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$
    - $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$
    - $\tilde{C}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$
    - $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
    - $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$
    - $h_t = o_t \odot \tanh(C_t)$
- **Backward Pass**: Uses BPTT to propagate the error from the output back through the sequence.

#### Integration
- Uses `pt.mleiria.utils.NormalizeUtils` for `sigmoid` and `tanh` activations.
- Leverages `breeze.linalg` for all matrix and vector operations.

### 3. Random Forest
A robust ensemble implementation of Decision Trees for both classification and regression tasks.

#### Core Architecture
Implemented in the `pt.mleiria.core.trees` package. It uses a modular approach separating the splitting criteria, the individual decision tree logic, and the ensemble management.

#### Key Components
- **`DecisionTree`**: A recursive binary tree implementation that splits data based on the best impurity reduction.
- **`RandomForest`**: Manages a collection of trees, implementing bagging (bootstrap aggregating) and random feature selection to improve generalization.
- **`SplitCriteria`**: Provides modular implementations for different tasks:
    - **Gini Impurity** & **Entropy** for classification.
    - **Mean Squared Error (MSE)** for regression.
- **`TreeConfig` & `RFConfig`**: Hyperparameter management for controlling tree depth, leaf size, and forest size.

#### Mathematical Concepts
- **Gini Impurity**: $G = 1 - \sum_{i=1}^{C} p_i^2$ (Used to measure the purity of a node in classification).
- **Entropy**: $H = -\sum_{i=1}^{C} p_i \log_2 p_i$ (Alternative impurity measure for classification).
- **Mean Squared Error (MSE)**: $MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \bar{y})^2$ (Used as the splitting criterion for regression).
- **Bagging**: Each tree is trained on a bootstrap sample (random sampling with replacement) of the original training set.
- **Feature Sampling**: Only a random subset of features is considered at each split to decorrelate the trees.

#### Integration
- Leverages `breeze.linalg` for efficient matrix handling.
- Implements parallel tree training using Scala `Future`s to utilize available CPU cores.

## Verification
The implementations are verified via:
- **LSTM**: Unit tests for forward pass dimensions and integration tests ensuring cost convergence on sequence prediction.
- **Random Forest**: 
    - Verification of impurity calculations.
    - Integration tests on synthetic separable data (Classification) and linear-ish data (Regression) to ensure accuracy and MSE reduction.

---
*This project is a part of the physical training data analysis suite.*
