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


## Pruning

**Definition:** Pruning is a technique used in decision trees to reduce the size of the tree by removing sections of the tree that provide little predictive power.

**Explanation:** Decision trees can grow to be very large and complex, especially when they have a deep hierarchy of nodes. Pruning helps to avoid overfitting by removing parts of the tree that are redundant or do not significantly improve the accuracy of predictions on the test data. There are two main types of pruning:
- **Pre-pruning**: Stops the tree construction early, before it reaches its maximum depth.
- **Post-pruning**: Allows the tree to grow fully and then prunes back the branches.

---

## Gini Index

**Definition:** The Gini index measures the impurity of a set of examples.

**Explanation:** In the context of decision trees, the Gini index is used to decide the optimal split at each node. It calculates the impurity of the data at a particular node. A lower Gini index indicates that a node contains predominantly samples from a single class, while a higher index means that the samples are distributed across different classes. The formula for Gini index at a node \( t \) is:
<p align="center"><img src="https://latex.codecogs.com/svg.latex?\text{Gini}(t)=1-\sum_{i=1}^{c}(p_i)^2"></p>
where \( c \) is the number of classes and \( p_i \) is the probability of class \( i \) at node \( t \).

---

## Entropy

**Definition:** Entropy measures the impurity or uncertainty of a set of examples.

**Explanation:** In decision trees, entropy is another metric used for determining the best split. It quantifies the amount of uncertainty in the data set. A lower entropy indicates that a node contains predominantly samples from a single class, while a higher entropy indicates that the samples are evenly distributed across different classes. The formula for entropy at a node \( t \) is:
<p align="center"><img src="https://latex.codecogs.com/svg.latex?\text{Entropy}(t)=-\sum_{i=1}^{c}p_i\log_2(p_i)"></p>
where \( p_i \) is the probability of class \( i \) at node \( t \).

---

## Information Gain

**Definition:** Information gain measures the effectiveness of a particular attribute in classifying the training data.

**Explanation:** Information gain is used to decide which attribute to split on at each step in building the decision tree. It quantifies the reduction in entropy or Gini index achieved by splitting the data based on a particular attribute. The attribute with the highest information gain is chosen as the splitting attribute at each node. The formula for information gain based on entropy is:
<p align="center"><img src="https://latex.codecogs.com/svg.latex?\text{Gain}(S,A)=\text{Entropy}(S)-\sum_{v\in\text{Values}(A)}\frac{|S_v|}{|S|}\text{Entropy}(S_v)"></p>
where \( S \) is the dataset, \( A \) is an attribute, \( \text{Values}(A) \) are the possible values of attribute \( A \), \( S_v \) is the subset of \( S \) for which attribute \( A \) has value \( v \).

---

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

# Gradient descent
Gradient descent is an optimization algorithm used to minimize the loss function in machine learning models. It iteratively adjusts the model parameters in the direction of the steepest descent of the loss function.
# Difference Between Gradient Descent and Stochastic Gradient Descent

Gradient Descent (GD) and Stochastic Gradient Descent (SGD) are both optimization algorithms used to minimize the cost function in machine learning models. They share the same goal but differ in how they update the model parameters.

## Gradient Descent (GD)

**Batch Gradient Descent (BGD):**

1. **Overview:**
   - Batch Gradient Descent computes the gradient of the cost function with respect to the parameters for the entire training dataset.
   - It updates the parameters by taking a step in the direction of the negative gradient.
   
2. **Steps:**
   1. Compute the gradient of the cost function for the entire dataset.
   2. Update the parameters using the computed gradient.
   3. Repeat until convergence.

3. **Advantages:**
   - Converges to the global minimum for convex functions.
   - More stable and accurate in each iteration.

4. **Disadvantages:**
   - Computationally expensive for large datasets.
   - Requires storing the entire dataset in memory.
   - Slow convergence, especially with large datasets.

## Stochastic Gradient Descent (SGD)

**Stochastic Gradient Descent (SGD):**

1. **Overview:**
   - Stochastic Gradient Descent computes the gradient of the cost function for a single training example at each iteration.
   - It updates the parameters more frequently, with higher variance in the updates.

2. **Steps:**
   1. Randomly shuffle the training dataset.
   2. For each training example:
      - Compute the gradient of the cost function for that example.
      - Update the parameters using the computed gradient.
   3. Repeat until convergence.

3. **Advantages:**
   - Faster convergence for large datasets.
   - Can handle large datasets that do not fit into memory.
   - Introduces noise in the parameter updates, which can help escape local minima.

4. **Disadvantages:**
   - High variance in the updates can lead to a less stable convergence.
   - May not converge to the exact global minimum but often good enough for practical purposes.

## Comparison

1. **Computation:**
   - **GD:** Uses the entire dataset for each update, making it computationally expensive.
   - **SGD:** Uses one training example per update, making it faster per update but noisier.

2. **Memory Usage:**
   - **GD:** Requires the entire dataset to be loaded into memory.
   - **SGD:** Can work with mini-batches or single examples, requiring less memory.

3. **Convergence:**
   - **GD:** Provides stable convergence but can be slow.
   - **SGD:** Faster convergence but with more variance in the updates.

4. **Suitability:**
   - **GD:** Better suited for smaller datasets.
   - **SGD:** Better suited for large datasets and online learning.

In summary, the main difference lies in how much data is used to compute the gradient at each step: GD uses the whole dataset, whereas SGD uses a single example or a small batch of examples. This leads to trade-offs in computation time, memory usage, and convergence stability.

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
## 4. Leaky ReLU Activation Function

The Leaky ReLU function is a variation of ReLU that allows a small, non-zero gradient when the input is negative. It’s defined as:

<p align="center"><img src="https://latex.codecogs.com/svg.latex?\text{LeakyReLU}(x)=\begin{cases}x&\text{if }x\geq0\\\alpha x&\text{if }x<0\end{cases}"></p>

**Properties:**
- **Range**: (-∞, ∞)
- **Advantages**:
  - Helps mitigate the dying ReLU problem by allowing a small gradient when the input is negative.
- **Disadvantages**:
  - The slope for negative inputs needs to be manually set and is a hyperparameter to tune.

## 5. Softmax Activation Function

The softmax function is typically used in the output layer of a neural network for classification problems. It converts logits into probabilities. It’s defined as:
<p align="center"><img src="https://latex.codecogs.com/svg.latex?\text{softmax}(x_i)=\frac{e^{x_i}}{\sum_{j}e^{x_j}}"></p>
**Properties:**
- **Range**: (0, 1) for each output, and the outputs sum to 1.
- **Advantages**:
  - Provides a probabilistic interpretation of the output, useful for multi-class classification.
- **Disadvantages**:
  - Computation can be expensive for large output spaces due to the normalization step.
