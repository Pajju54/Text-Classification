# Naive Bayes Classifier for Text Classification

This repository contains the implementation of a Naive Bayes Classifier for text classification. The classifier is implemented in Python and is designed to classify sentences into various categories.

## **Project Overview**
Naive Bayes is a simple yet effective classification algorithm widely used for text-based tasks like spam detection and document categorization. It works by applying Bayes' Theorem and assuming that features are independent of each other. This classifier predicts the likelihood of a category given certain features (e.g., words in a sentence).

### **Project Goals**
The main objectives of this project are to:
1. Preprocess the dataset by handling incorrect and missing labels and removing stop words.
2. Implement the core functions of a Naive Bayes Classifier:
   - `preprocess()`
   - `fit()`
   - `predict()`
3. Test the classifier against predefined and hidden test cases.

## **File Structure**
- **naive_bayes.py**: 
  - This is the main Python file where the NaiveBayesClassifier class is implemented. It contains three key functions:
    1. `preprocess(sentences, categories)`: Preprocesses the dataset by removing incorrect and missing labels and eliminating stop words.
    2. `fit(X, y)`: Trains the Naive Bayes Classifier using the provided training data.
    3. `predict(X, class_probs, word_probs, classes)`: Predicts the category for the given test data using the trained classifier.

- **naive_bayes_test.py**:
  - Contains predefined test cases to evaluate the accuracy of the Naive Bayes Classifier implementation.

- **data**:
  - This directory contains the dataset used for training and testing the classifier, including sentences and their corresponding categories.

## **How to Run**

1. Clone the repository:
   ```bash
   git clone https://github.com/Pajju54/naive-bayes-classifier.git
   cd naive-bayes-classifier
   ```
2. Ensure that the Python environment is set up with the necessary dependencies (if any).

3. Run the test file to evaluate the classifier:
    ```bash
    python naive_bayes_test.py
    ```

The program will automatically test your implementation against the provided test cases and display the results.

# Functions Implemented

## preprocess(sentences, categories)
Purpose: Cleans the data by removing incorrect labels (wrong_label), handling missing labels (None), and eliminating stop words for better classification performance.

## fit(X, y)
Purpose: Trains the classifier using the input data (X - sentences, y - corresponding categories). This function calculates the prior probabilities of each class and the likelihood of words given each class.

## predict(X, class_probs, word_probs, classes)
Purpose: Given new input data, this function predicts the most probable category using the trained model, class probabilities, and word probabilities.

# Evaluation Criteria
1. Accurate implementation of the NaiveBayesClassifier class.
2. Successfully passing all predefined and hidden test cases provided in the naive_bayes_test.py file.

# Author
Name: Prajwal M Joshi
