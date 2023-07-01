# ICD Algorithm Implementation

## Overview
This is an implementation of the ICD algorithm for classification tasks. The ICD algorithm is an unsupervised learning algorithm that finds prototypes for each class in the dataset. It can be applied to various classification problems, such as image or text classification.

## Requirements
To run this code, you will need the following libraries and dependencies:
- Python 3.x
- NumPy
- Matplotlib
- scikit-learn
- Keras

## Usage
To use the program, follow these steps:
1. Set up your Python environment and install the required libraries.
2. Import the necessary libraries and the ICD class from the `icd.py` file.
3. Initialize the ICD class with the desired dataset and parameters.
4. Call the `initialize_prototypes()` and `update_prototypes()` methods to train the ICD algorithm on the dataset.
5. Evaluate the performance of the model using the `evaluate_model()` method.
6. Visualize the prototypes using the `visualize_prototypes()` method.

## Examples
The `main.py` file contains examples of how to use the ICD algorithm for the Iris and MNIST datasets. Check the file for code snippets and explanations on how to train, evaluate, and visualize the ICD algorithm's performance.

## API Reference
The following classes, methods, and functions are available in the `icd.py` file:
- `ICD`: The main class implementing the ICD algorithm.
- `initialize_prototypes()`: Initializes the prototypes with the first example of each class.
- `update_prototypes()`: Updates the prototypes based on the training data.
- `euclidean_distance()`: Computes the Euclidean distance between two points.
- `predict_class()`: Predicts the class of a given example.
- `evaluate_model()`: Evaluates the performance of the model on the test set.
- `visualize_prototypes()`: Visualizes the prototypes using dimensionality reduction.

Please refer to the inline documentation in the `icd.py` file for a more detailed explanation of each method's purpose, input parameters, and return values.

## Tutorials
For step-by-step tutorials on how to use the ICD implementation for various tasks, please refer to the provided examples in the `main.py` file. The examples demonstrate how to load datasets, initialize the ICD algorithm, train the classifier, evaluate its performance, and visualize the resulting prototypes.

## Troubleshooting
Some common issues and their solutions when using the ICD implementation are as follows:
- Low performance or accuracy: If the ICD algorithm is not performing well on a specific dataset, try adjusting the distance_limit parameter to control the creation of new prototypes. Also, ensure that the data is properly preprocessed and normalized before training.

- Memory issues or slow performance: When working with large datasets like MNIST, you might face memory constraints or slow processing times. You can reduce the size of the dataset by using a smaller subset, which can be specified using the subset_size parameter when initializing the ICD class.

- Visualization issues: If you encounter problems while visualizing the prototypes, make sure you have the correct dimensionality reduction algorithm installed and configured (e.g., PCA from scikit-learn). Additionally, ensure that the dataset dimensions are compatible with the chosen dimensionality reduction algorithm.

- Library or dependency issues: If you encounter issues related to missing libraries or dependencies, double-check that you have installed all the required libraries listed in the "Requirements" section.


For any other issues or questions, please refer to the inline documentation in the icd.py file or the provided examples in the main.py file.
