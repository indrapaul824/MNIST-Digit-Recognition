<h1 align="center">MNIST-Digit-Recognition</h1>

Implemented various Machine Learning and Deep Learning Algorithms on the famous digit recognition problem using the **MNIST** (Mixed National Institute of Standards and Technology) database.

<h3>Dataset:</h3>
The MNIST database contains binary images of handwritten digits commonly used to train image processing systems. The digits were collected from among Census Bureau employees and high school students. The database contains 60,000 training digits and 10,000 testing digits, all of which have been size-normalized and centered in a fixed-size image of 28 × 28 pixels. Many methods have been tested with this dataset and in this project, I experimented with the task of classifying these images into the correct digit using both Machine Learning and Deep Learning approaches.


<h3>Setup:</h3>

I used Python's **NumPy** numerical library for handling arrays and array operations; used **matplotlib** for producing figures and plots.

1. Note on software: I used python 3.8 augmented with the NumPy numerical toolbox, the matplotlib plotting toolbox. In this project, I also used the scikit-learn package, which you could install by `conda install scikit-learn` or `pip install sklearn`.

*Download .zip file* or *clone the repo* into a working directory. The folder contains the various data files in the Dataset directory, along with the following python files:
> <h4>Machine Learning Approach:</h4>
  * part1/linear_regression.py where I've implemented linear regression
  * part1/svm.py where I've implemented support vector machine
  * part1/softmax.py where I've implemented multinomial regression
  * part1/features.py where I've implemented principal component analysis (PCA) dimensionality reduction
  * part1/kernel.py where I've implemented polynomial and Gaussian RBF kernels
  * part1/main.py where I've used the above mentioned modules in the MNIST dataset for this part of the project
