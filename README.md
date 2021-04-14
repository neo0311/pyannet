# PyANNET

A Python package for deep multi-output regressions and data handling.

## Prerequisites

*pip*, *numpy*, *matplotlib*

## Installation

* Clone the repository

`git clone https://github.com/neo0311/pyannet.git`

* Run in terminal

`python -m pip install -e ".[dev]"`

## Functionality

### Data Handling

#### Sampling methods:

Various methods for multidimensional sampling are available.

1. **Latin Hypercube Sampling**
2. **Quasi-Monte Carlo Sampling**
   1. Halton
   2. Hammersley

#### Data Preparation

1. **Data Transformation**
   1. Min-max normalisation
   2. Z-score normalisation
2. **Test-train split**
3. **Read training data from file**
4. **Shuffle training data**

### Neural Netwok

1. **Architecture** - Feed Forward Neural Network
2. **Weight Initialization** - Xavier and He methods
3. **Activation Functions**
   1. Swish
   2. Linear
   3. Relu
   4. Leaky Relu
   5. Sigmoid
4. **Backpropagation**
   * **Cost Function**
     * Mean Square Error (MSE)
   * **Algorithm**
     * Stochastic Gradient Descent
     * Batch Gradient Descent
     * Mini-Batch Gradient Descent
   * **Optimizers**
     * Adam optimizer
5. **Metrics for evaluation**
   1. Mean Squared Error (MSE)
   2. Mean Absolute Error (MAE)
   3. Mean Absolute Percentage Error (MAPE)
   4. Coefficient of determination (R2)
   5. Root Mean Squared Error (RMSE)
   6. Mean Squared Log Error (MSLE).

## Testing

### Tests Available

1. Data Handling
   * Extensive unit tests
   * Dimension space checking
2. Neural Network
   * Extensive unit tests
   * Gradient Checking
   * Forward propagation with known inputs, weights and outputs

Run

`pytest tests -v`

to run all the tests for `pyannet`.
