# AIML-001 Level 1 Report

## Task 1: Linear and Logistic Regression

**Linear Regression** is a supervised learning algorithm that predicts a continuous value by fitting a line to the data. I built a model to predict the likelihood of diabetes using the diabetes dataset from `sklearn`, as the Boston and California datasets had issues.

**Logistic Regression** is a supervised learning algorithm that predicts a binary outcome by modeling probability with a logistic function. Using `sklearn.linear_model.LogisticRegression`, I classified Iris flowers into two groups based on sepal length and width.

Overall, this task helped me dive into supervised learning models, making predictions and classifications.

[Linear Regression](https://github.com/bakeyed/AIML-001-Level-2/blob/main/Linear-Regression.ipynb)

[Logistic Regression](https://github.com/bakeyed/AIML-001-Level-2/blob/main/Logistic-Regression.ipynb)

## Task 2: Matplotlib and Data Visualization

**Matplotlib** is a Python library for creating visualizations, especially useful in data analysis for plotting data trends, distributions, and comparisons. I explored several plot types—like bar, stacked, histogram, violin, pie, line, and area plots—and learned how to use labels, legends, and subplots. This helped me understand each plot’s purpose in presenting different data insights.

This image includes some of the plots I made.

[Plots](https://github.com/bakeyed/AIML-001-Level-2/blob/main/Plots.ipynb)

## Task 3: Numpy

**NumPy** is a powerful library for numerical computations in Python, widely used for handling arrays and mathematical operations efficiently. I used NumPy to generate an array by repeating a smaller array across dimensions, and I created an array with element indexes arranged in ascending order. This practice helped me get familiar with array manipulation and indexing in NumPy.

[Numpy](https://github.com/bakeyed/AIML-001-Level-2/blob/main/marvel-numpy.py)

## Task 4: Metrics and Performance Evaluation

Metrics and Performance Evaluation are essential in machine learning to assess how well models perform.

**Regression metrics** like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared assess how well models predict continuous values.

**Classification metrics** such as accuracy, precision, recall, F1 score, and AUC-ROC evaluate the effectiveness of models in predicting discrete classes.

This task helped me evaluate my model and consider different optimization techniques to help solve issues.

## Task 5: Linear and Logistic Regression from Scratch

In this task, I implemented Linear and Logistic Regression models from scratch, deepening my understanding of their functionality.

**Linear Regression** employs the familiar formula `y=mx+b` , where y represents the predicted value, xxx is the input parameter, m is the slope, and b is the y-intercept.

**Logistic Regression** predicts a binary outcome using the sigmoid function, which helps separate classes. While commonly used for binary classification, it can also be adapted for multiclass problems. This implementation enhanced my grasp of both models and their applications in data analysis.

## Task 6: K-Nearest Neighbor Algorithm

KNN classifies data by identifying the 'k' closest examples and using majority voting to determine the predicted class. This approach effectively distinguishes between different iris species based on sepal and petal measurements.

[KNN](https://github.com/bakeyed/AIML-001-Level-2/blob/main/KNN.ipynb)

## Task 7: An elementary step towards understanding Neural Networks

[Link to my Blog](https://github.com/bakeyed/AIML-001-Level-2/blob/main/Blog.md)

## Task 8: Mathematics behind machine learning

In this task, I deepened my understanding of the mathematical constructs fundamental to machine learning algorithms.

1.  **Curve Fitting**: I modeled a curve fitting for the cardioid function using Desmos.
2.  **Fourier Transforms**: I watched a 3Blue1Brown video on Fourier transforms. Although I didn't have access to MATLAB, I observed its application through a YouTube video. I'm also learning more about Fourier transforms in my M3 course.

## Task 9: Data Visualization for Exploratory Data Analysis

## Plotly

Plotly is an open-source, interactive graphing library for Python that allows users to create a variety of charts, including 3D graphs, financial charts, and statistical charts. This makes visualizations more dynamic and facilitates exploratory data visualization
I created an interactive bubble chart on the different types of pet owners in Bangalore. This chart is more interactive than a simple plot, allowing users to engage with the data and explore trends more effectively. The interactivity enhances the overall experience, making it easier to understand and analyze the data presented.
[Plotly](https://github.com/bakeyed/AIML-001-Level-2/blob/main/Plotly.ipynb)

## Task 10: An introduction to Decision Trees

A Decision Tree is a supervised learning algorithm used for regression or classification tasks. It structures conditional statements in a hierarchy to predict outcomes based on input features, making it easy to interpret decisions.

**Components of a Decision Tree**:

- **Root Node:** The topmost node that represents the entire dataset and splits into branches based on the best feature.
- **Branches:** Lines that connect nodes, representing outcomes of decisions made at a parent node, leading to either another node or a leaf node.
- **Internal/Decision Nodes:** Represent features used for decision-making. Each internal node splits the data based on a specific condition related to that feature.
- **Leaf Nodes (Terminal Nodes):** End points of the tree representing final outcomes or predictions after all decisions have been made.

For this task, I used the Iris dataset once again, which involves predicting the species of iris flowers based on their features.
[Decision Trees](https://github.com/bakeyed/AIML-001-Level-2/blob/main/Decision-Trees.ipynb)

## Task 11: SVM

Support Vector Machines (SVM) are supervised learning methods used to create a non-probabilistic linear model. In this approach, each data value is assigned to one of two classes with the aim of maximizing the difference between those classes. The data points are represented as vectors, and an optimal hyperplane is selected to enhance the separation between the two classes while also regularizing the loss.
I used SVM to predict the probability of breast cancer using a dataset from sklearn. A crucial aspect of improving the model’s performance is normalization.
Normalization adjusts the scales of features, ensuring that each one contributes equally to the distance calculations performed by the SVM. This process enhances prediction accuracy and helps the model converge more quickly. By scaling features to a consistent range, normalization facilitates the optimal selection of the hyperplane.
I plan to explore normalization techniques further in future work to understand their impact on model performance and accuracy.

[SVM](https://github.com/bakeyed/AIML-001-Level-2/blob/main/SVM.ipynb)
