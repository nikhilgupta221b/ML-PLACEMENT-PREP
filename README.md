# ML-SHORTS
## Artificial Intelligence (AI)
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. AI systems can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.

## Machine Learning (ML)
Machine Learning (ML) is a subset of AI that involves the use of algorithms and statistical models to enable computers to improve their performance on a task through experience. ML systems learn from data and make predictions or decisions without being explicitly programmed to perform the task.

## Linear Regression
Linear Regression is a statistical method for modeling the relationship between a dependent variable and one or more independent variables. It assumes a linear relationship between the input variables (features) and the output variable, and it tries to find the best-fitting straight line (regression line) through the data points.

## Logistic Regression
Logistic Regression is a statistical method for binary classification problems. It models the probability that a given input belongs to a certain class. It uses a logistic function (sigmoid function) to map the predicted values to probabilities, which are then used to classify the inputs into one of two categories.

## Random Forest Classifier
Random Forest Classifier is an ensemble learning method used for classification (and regression) tasks. It operates by constructing multiple decision trees during training and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random forests are known for their accuracy and ability to handle large datasets with higher dimensionality.

## Decision Tree
Decision Tree is a supervised learning algorithm used for both classification and regression tasks. It works by recursively splitting the dataset into subsets based on the value of an input feature, creating a tree-like model of decisions. Each internal node represents a test on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label (in classification) or a continuous value (in regression).

## Support Vector Machine (SVM)
Support Vector Machine (SVM) is a supervised learning algorithm used for classification and regression tasks. It works by finding the hyperplane that best separates the classes in the feature space. SVM aims to maximize the margin between the closest points of the classes (support vectors), making it effective for high-dimensional spaces and when the number of dimensions exceeds the number of samples.

## Neural Networks
Neural Networks are a series of algorithms that attempt to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. They are a key technology in machine learning and artificial intelligence, particularly in deep learning.

Neural networks consist of layers of interconnected nodes, or "neurons," where each connection is associated with a weight. These networks typically include:

- Input Layer: This layer receives the initial data.
- Hidden Layers: One or more layers between the input and output layers where the computation happens. Each neuron in a hidden layer takes a weighted sum of the inputs, applies an activation function, and passes the result to the next layer.
- Output Layer: This layer produces the final output of the network.

Key concepts in neural networks include:

- Weights: Parameters within the network that transform input data within the network's layers.
- Biases: Additional parameters that allow the activation function to be shifted to the left or right, which helps the model fit the data better.
- Activation Functions: Functions applied to the input of each neuron to introduce non-linearity into the network, enabling it to model complex relationships. Common activation functions include ReLU (Rectified Linear Unit), sigmoid, and tanh.
- Training: The process of adjusting the weights and biases of the network using a dataset, typically through a method called backpropagation, which uses gradient descent to minimize the error in the network's predictions.
Neural networks are used in a wide range of applications, from image and speech recognition to natural language processing and even playing games, due to their ability to learn from and adapt to complex data patterns.

## Actuvation Functions:

# Activation Functions

## 1. Sigmoid Activation Function

The sigmoid function maps any input to a value between 0 and 1. It’s defined as:

<p align="center"><img src="https://latex.codecogs.com/svg.latex?\sigma(x)=\frac{1}{1+e^{-x}}"></p>

**Properties:**
- **Range**: (0, 1)
- **Advantages**:
  - Outputs are bounded, making it useful for probabilities.
- **Disadvantages**:
  - Can cause vanishing gradient problem as gradients get very small for large positive or negative inputs.
  - Outputs are not zero-centered, which can slow down convergence.

## 2. Tanh (Hyperbolic Tangent) Activation Function

The tanh function maps any input to a value between -1 and 1. It’s defined as:

<p align="center"><img src="https://latex.codecogs.com/svg.latex?\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}"></p>

**Properties:**
- **Range**: (-1, 1)
- **Advantages**:
  - Outputs are zero-centered, which helps in faster convergence.
  - Less prone to the vanishing gradient problem compared to the sigmoid function.
- **Disadvantages**:
  - Can still cause vanishing gradient problem, though less severe than sigmoid.

## 3. ReLU (Rectified Linear Unit) Activation Function

The ReLU function is defined as:

<p align="center"><img src="https://latex.codecogs.com/svg.latex?\text{ReLU}(x)=\max(0,x)"></p>

**Properties:**
- **Range**: [0, ∞)
- **Advantages**:
  - Computationally efficient as it involves simple thresholding at zero.
  - Helps mitigate the vanishing gradient problem by not saturating for large positive values.
- **Disadvantages**:
  - Can cause dying ReLUs problem where neurons can get stuck during training if they enter a regime where they always output zero.

## 4. Softmax Activation Function

The softmax function is typically used in the output layer of a neural network for classification problems. It converts logits into probabilities. It’s defined as:

<p align="center"><img src="https://latex.codecogs.com/svg.latex?\text{softmax}(x_i)=\frac{e^{x_i}}{\sum_{j}e^{x_j}}"></p>

**Properties:**
- **Range**: (0, 1) for each output, and the outputs sum to 1.
- **Advantages**:
  - Provides a probabilistic interpretation of the output, useful for multi-class classification.
- **Disadvantages**:
  - Computation can be expensive for large output spaces due to the normalization step.
