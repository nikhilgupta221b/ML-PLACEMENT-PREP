# ML-SHORTS
## Artificial Intelligence (AI)
Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans. AI systems can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.

## Machine Learning (ML)
Machine Learning (ML) is a subset of AI that involves using algorithms and statistical models to enable computers to improve their performance on a task through experience. ML systems learn from data and make predictions or decisions without being explicitly programmed to perform the task.

## Linear Regression
Linear Regression is a statistical method for modeling the relationship between a dependent variable and one or more independent variables. It assumes a linear relationship between the input variables (features) and the output variable, and it tries to find the best-fitting straight line (regression line) through the data points.

## Logistic Regression
Logistic Regression is a statistical method for binary classification problems. It models the probability that a given input belongs to a certain class. It uses a logistic function (sigmoid function) to map the predicted values to probabilities, which are then used to classify the inputs into one of two categories.

# Correlation in Machine Learning

**Correlation** in machine learning refers to a statistical measure that describes the extent to which two variables move in relation to each other. It helps identify relationships between features and can be crucial for feature selection and understanding data relationships.

## Types of Correlation

1. **Positive Correlation**: As one variable increases, the other variable tends to increase. For example, height and weight often have a positive correlation.
2. **Negative Correlation**: As one variable increases, the other variable tends to decrease. For example, the number of hours spent studying and the number of mistakes on a test often have a negative correlation.
3. **No Correlation**: There is no discernible pattern or relationship between the variables. 

## Common Correlation Coefficients

- **Pearson Correlation Coefficient**: Measures the linear relationship between two continuous variables. It ranges from -1 (perfect negative correlation) to +1 (perfect positive correlation). A value of 0 indicates no linear correlation.
  
- **Spearman Rank Correlation**: Measures the strength and direction of the monotonic relationship between two ranked variables.

- **Kendall Tau Correlation**: Measures the ordinal association between two measured quantities.

## Visualizing Correlation

Visual representations help understand and interpret the strength and direction of relationships between variables. Below is an image showing the formula for Pearson's correlation coefficient:

<div style="text-align: center;">
  <img src="https://www.gstatic.com/education/formulas2/553212783/en/correlation_coefficient_formula.svg" alt="Correlation Coefficient Formula">
</div>


In the formula:

- \( r \) is the Pearson correlation coefficient.
- \( x_i \) and \( y_i \) are the individual sample points indexed with \( i \).
- \( \bar{x} \) and \( \bar{y} \) are the means of the x and y samples, respectively.

Correlation matrices are especially useful for examining relationships in datasets with many features.

## Importance of Correlation in Machine Learning

- **Feature Selection**: Correlation helps in selecting features that have strong relationships with the target variable while avoiding those that are highly correlated with each other to reduce multicollinearity.
- **Model Interpretation**: Understanding feature correlations can provide insights into model behavior and the importance of features.
- **Data Preprocessing**: Highly correlated features might be redundant and can be removed to simplify models without losing significant predictive power.

## Key Considerations

- **Correlation Does Not Imply Causation**: Just because two variables are correlated does not mean that one causes the other.
- **Outliers**: Correlation coefficients can be sensitive to outliers. It’s important to check data for outliers before interpreting correlation values.
- **Non-Linear Relationships**: Correlation coefficients like Pearson’s measure only linear relationships. They may not capture non-linear patterns in data.

## Example Code for Calculating Correlation in Python

```python
import pandas as pd

# Sample data
data = {'Feature1': [1, 2, 3, 4, 5],
        'Feature2': [2, 4, 6, 8, 10],
        'Feature3': [5, 4, 3, 2, 1]}

df = pd.DataFrame(data)

# Calculate correlation matrix
correlation_matrix = df.corr()

print(correlation_matrix)
```

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


Zero-centered outputs refer to a scenario where the outputs of a function, model, or system are distributed around a mean of zero. This concept is often used in the context of machine learning, signal processing, and data normalization.

Why Zero-Centered Outputs Matter
Improved Training Stability: In machine learning, particularly in neural networks, having inputs and outputs centered around zero can improve the stability and speed of training. It helps in avoiding issues where gradients can vanish or explode during backpropagation, making the learning process more efficient.

Better Weight Updates: When outputs are zero-centered, weight updates during training (like in gradient descent) tend to be more balanced. This means that the weights are adjusted more symmetrically, which can lead to a faster convergence to the optimal solution.

Symmetry in Activation Functions: Many activation functions (like the hyperbolic tangent 
tanh
⁡
tanh or certain variations of ReLU) are symmetric around zero. Zero-centered inputs help maintain this symmetry, ensuring that the activations are also centered around zero, which is desirable for balanced and effective learning.

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

The vanishing gradient problem is a phenomenon that occurs during the training of deep neural networks, where the gradients of the loss function with respect to the model's parameters become very small. This causes the weights in the earlier layers of the network to update very slowly, effectively making it difficult for the network to learn and improve during training.

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

The dying ReLU problem refers to a situation in neural networks where a significant number of neurons using the ReLU (Rectified Linear Unit) activation function become inactive, meaning they only output zero for any input. Once a neuron becomes inactive in this way, it effectively "dies" because it stops contributing to the learning process, and the network may lose the ability to learn from certain parts of the data.

## 5. Softmax Activation Function

The softmax function is typically used in the output layer of a neural network for classification problems. It converts logits into probabilities. It’s defined as:
<p align="center"><img src="https://latex.codecogs.com/svg.latex?\text{softmax}(x_i)=\frac{e^{x_i}}{\sum_{j}e^{x_j}}"></p>
**Properties:**
- **Range**: (0, 1) for each output, and the outputs sum to 1.
- **Advantages**:
  - Provides a probabilistic interpretation of the output, useful for multi-class classification.
- **Disadvantages**:
  - Computation can be expensive for large output spaces due to the normalization step.

## Bagging vs. Boosting

### Bagging (Bootstrap Aggregating)

**Definition**: Bagging is an ensemble learning technique that combines the predictions of multiple base models to improve overall accuracy and reduce variance. It works by training multiple models on different subsets of the data and then aggregating their predictions.

**How It Works**:
1. **Bootstrap Sampling**: Create multiple subsets of the training data by sampling with replacement.
2. **Model Training**: Train a separate model on each subset.
3. **Aggregation**: Combine predictions from all models. For regression, this typically means averaging the predictions; for classification, it might involve voting.

**Advantages**:
- Reduces variance and overfitting.
- Useful when the base model is unstable (e.g., decision trees).

**Disadvantages**:
- Can be computationally expensive.
- May not improve performance if base models are too similar.

**Example**:
- **Random Forest** is a popular bagging algorithm that uses decision trees as base models.

### Boosting

**Definition**: Boosting is an ensemble learning technique that builds models sequentially, each new model focusing on the errors made by the previous ones. The idea is to convert weak learners into a strong learner.

**How It Works**:
1. **Initial Model**: Train an initial model on the full dataset.
2. **Error Correction**: Train subsequent models to correct the errors of the previous models, usually by weighting the errors.
3. **Combination**: Combine the predictions of all models, often by weighted averaging.

**Advantages**:
- Reduces bias and variance.
- Often leads to better performance on the training data.

**Disadvantages**:
- Can be sensitive to noisy data and outliers.
- More prone to overfitting compared to bagging.

**Example**:
- **Gradient Boosting** and **AdaBoost** are well-known boosting algorithms.

## Feature Engineering Techniques in Machine Learning

Feature engineering involves creating new features or modifying existing ones to improve the performance of machine learning models. Here are some key techniques:

### 1. **Feature Scaling**

**Definition**: Adjusting the range of features to improve model performance and convergence speed.

**Techniques**:
- **Normalization**: Scales features to a range of [0, 1].
- **Standardization**: Scales features to have a mean of 0 and a standard deviation of 1.

**Example**:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
