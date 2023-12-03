# Introduction in Data Science

This repository consists of 6 homeworks I did at university for the course "Introduction to Data Science".

## Homework 1

This homework consists of 8 exercises covering various Python programming concepts.

### Exercise 1
Calculate the difference between the sum of the 3 largest numbers and the product of the 2 smallest numbers in a given list.

### Exercise 2
Determine unique words and their occurrences in a list of words.

### Exercise 3
Find the three numbers in a list that, when multiplied together, give the maximum product.

### Exercise 4
Given a list of numbers and a natural number n, determine triplets a, b, c with a < b in the list, where a^n + b^n = c^n.

### Exercise 5
Translate a sentence into its "birdified" version, where vowels become a vowel + 'P' + vowel, and vice versa.

### Exercise 6
Explore a dictionary with student names as keys and lists of favorite books as values. Perform various queries, including finding students with specified books, pairs with the same preferences, and more.

### Exercise 7
Given a dictionary of persons and their availability times, determine intervals in which all persons are available or intervals with the maximum number of participants.

### Exercise 8
Analyze a list of dictionaries representing persons. Determine average, minimum, and maximum wages; find persons of a given age within an interval; identify the most preferred animal; and count the number of males and females.

---

**Note:** Detailed solutions for each exercise can be found in their respective directories.


## Homework 2

This homework consists of 10 exercises covering various Python programming and NumPy concepts.

### Exercise 1
Write a function to determine whether a number is equal to the sum of its divisors minus the number itself. Display numbers with this property up to a given limit.

### Exercise 2
Write a function to check if a string is palindromic, with an option to ignore spaces. Handle exceptions for invalid input.

### Exercise 3
Create a function to compare numbers in two text files, returning common numbers and the maximum difference between corresponding numbers.

### Exercise 4
Implement functions to calculate the sum of digits of divisors and explore a "black hole" function based on repeated calls.

### Exercise 5
Write a function to check if one dictionary is contained in another.

### Exercise 6
Create functions to find the first-order difference for a vector and a matrix along rows or columns.

### Exercise 7
Implement a function to find positions of local maxima in a NumPy vector.

### Exercise 8
Develop a function to compute windows of a given length with a specified dilation for a NumPy vector.

### Exercise 9
Download and preprocess the Wine dataset, introducing NaN values. Create a function to identify columns with NaN values and fill them with a default value.

### Exercise 10
Write a function to determine the positions of the k nearest neighbors of a given value in a vector.

---

**Note:** Detailed solutions for each exercise can be found in their respective directories.


## Homework 3

This homework consists of four exercises covering polynomial functions, linear regression, data visualization, and interpolation.

### Exercise 1
**Objective:** Apply the gradient descent algorithm to minimize a predefined polynomial function of degree 3.

- Define a polynomial function $f:\mathbb{R} \rightarrow \mathbb{R}$.
- Use the gradient descent algorithm with at least two ipywidgets controls.
- Perform at least 10 iterations, marking the successive positions on the graph.

### Exercise 2
**Objective:** Explore linear regression and mean squared error for a set of coordinate points.

- Generate a list of $n=100$ pairs of values $\{x_i, y_i\}$ in the range [-20, 10).
- Display these values on a graph along with a controllable linear function $y=a \cdot x+b$.
- Create a histogram of the distance between the coordinate points and the line.
- Display the mean squared error (MSE) based on the line's coefficients.

### Exercise 3
**Objective:** Visualize numeric columns of the iris dataset based on user choices.

- Load the `iris.data` file and skip the header using numpy or pandas.
- Display the chosen numeric columns in a 2D graph.
- Allow user choices for the columns using ipywidgets.

### Exercise 4
**Objective:** Generate random points from a function and perform interpolation using scipy.

- Generate $n$ pairs of random points from a function with random noise.
- Choose 5 interpolation methods from scipy.interpolate.
- Plot the interpolated values, allowing user control for $n$ and interpolation method.

---

**Note:** Detailed solutions and code for each exercise can be found in their respective directories.


## Homework 4

This homework is focused on applying missing value imputation and evaluating classification models using scikit-learn.

### Exercise 1
**Objective:** Apply a missing value imputation method where applicable and document the method used.

- Implement missing value imputation for relevant datasets.
- Document the imputation method employed in the Jupyter notebook.

### Exercise 2
**Objective:** Evaluate 5 classification models on various datasets using fixed hyperparameter values.

- Apply 5 classification models from scikit-learn to each dataset.
- Report accuracy, precision, recall, and F1 score using 5-fold cross-validation.
- Provide mean results for both training and test folds.

### Exercise 3
**Objective:** Report model performance using 5-fold cross-validation with optimal hyperparameters.

- Evaluate each model's performance using 5-fold cross-validation.
- Search for optimal hyperparameters using 4-fold cross-validation in each of the 5 runs.
- Report the average performance over the 5 runs for each model.

### Exercise 4
**Objective:** Document each model used in the Jupyter notebook.

- Create documentation for each classification model used.
- If the same algorithm is applied to multiple datasets, make a separate section with algorithm documentation and references.

---

**Note:** Detailed solutions, code, and model documentation for each exercise can be found in their respective directories.


## Homework 5

This homework involves applying regression models to different datasets using scikit-learn and generating a comprehensive report.

### Datasets

1. [CPU Computer Hardware](https://archive.ics.uci.edu/ml/datasets/Computer+Hardware)
2. [Boston Housing](http://archive.ics.uci.edu/ml/machine-learning-databases/housing/)
3. [Wisconsin Breast Cancer](http://www.dcc.fc.up.pt/~ltorgo/Regression/DataSets.html)
4. [Communities and Crime](http://archive.ics.uci.edu/ml/datasets/communities+and+crime)

### Regression Models

For each dataset, apply at least 5 regression models from scikit-learn. Perform grid search (cv=3) and random search (n_iter specified) for hyperparameter tuning. The metric for optimization is mean squared error. Evaluate the models using 5-fold cross-validation and report mean absolute error, mean squared error, and median absolute error.

### Intermediate Report

The results will be stored in a dataframe. Due to implementation specifics in scikit-learn, the scores are presented as negative values. The values will be converted to the positive range, and maximum and minimum values will be marked.

### Final Report

A final report will be generated in HTML or PDF format. The report will include the dataset name, the dataframe object, and maintain color markings. 

### Grading

1. Default points.
2. Optimization and quantification of model performance: 3 points for each dataset-model combination = 60 points.
3. Model Documentation: Number of models * 2 points = 10 points. Each model must have a description of at least 20 lines, an associated image, and at least 2 bibliographic references.
4. 10 points: Export in HTML or PDF format.

---

**Note:** Detailed solutions, code, and model documentation for each dataset can be found in their respective directories.


## Homework 6

In this homework, text-based classification or regression problems are solved using datasets obtained from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php?format=&task=&att=&area=&numAtt=&numIns=&type=text&sort=taskDown&view=table). Various encoding methods such as Bag-of-Words (BOW), n-grams, word2vec, or other suitable techniques are employed. The prediction models are selected from the scikit-learn library, and NLTK is used for preprocessing.

### Datasets Investigated:

1. [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
2. [Sentence Classification Data Set](https://archive.ics.uci.edu/ml/datasets/Sentence+Classification#)
3. [Sentiment Labelled Sentences Data Set](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences)
4. [Victorian Era Authorship Attribution Data Set](https://archive.ics.uci.edu/ml/datasets/Victorian+Era+Authorship+Attribution)
5. [Amazon Commerce reviews set Data Set](https://archive.ics.uci.edu/ml/datasets/Amazon+Commerce+reviews+set)
6. [Farm Ads Data Set](https://archive.ics.uci.edu/ml/datasets/Farm+Ads)

### Investigation Steps:

1. **Dataset Description (2 x 0.5 points):** Provide a detailed description of each dataset in Romanian, covering content, origin, problem, etc.

2. **Exploratory Analysis (2 x 1 point):** Conduct exploratory analysis using Python code. This includes distribution analysis of classes or continuous output values, both numerically and graphically. Additionally, perform statistics on texts such as length distribution, most frequent words, and clustering.

3. **Data Pre-processing (2 x 0.5 points):** Pre-process the data, explaining the methods used and their effects on the input data. Show the impact of pre-processing steps on the dataset, including changes in the number of documents, vocabulary dynamics, resulting features, etc.

4. **Classification or Regression (2 x 4 x 0.5 points):** Describe the models considered for classification or regression in Romanian. Explain the hyperparameter search strategy and present the results in a tabular format.

### References:

- [Working With Text Data](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- [Text Classification with Python and Scikit-Learn](https://stackabuse.com/text-classification-with-python-and-scikit-learn/)
- [How to Prepare Text Data for Machine Learning with scikit-learn](https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/)

---

**Note:** Detailed solutions, code, and model documentation for each dataset can be found in their respective directories.
