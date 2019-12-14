Table of Contents
=================
   * [Business Problem](#business-problem)
   * [Notebooks](#notebooks)
   * [Text Data Processing](#text-data-processing)
   * [Visualization](#visualization)
   * [Modelling](#modelling)
   * [Model Evaluation](#model-evaluation)
   * [Model Explanation using lime](#model-explanation-using-lime)

# Business Problem
We are given large number of Wikipedia comments which have been labeled by human raters for toxic behavior.
The types of toxicity are: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.
We should create a model which predicts a probability of each type of toxicity for each comment.

# Notebooks
|  Notebook | Rendered   | Description  |  Author |
|---|---|---|---|
| a01_text_data_processing.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_Toxic_Comments/blob/master/notebooks/a01_text_data_processing.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_Toxic_Comments/blob/master/notebooks/a01_text_data_processing.ipynb)  | Creating new text features  | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| a02_text_data_eda_and_visualization.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_Toxic_Comments/blob/master/notebooks/a02_text_data_eda_and_visualization.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_Toxic_Comments/blob/master/notebooks/a02_text_data_eda_and_visualization.ipynb)  | Class distribution, word cloud, etc  | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| a03_text_data_modelling_logistic_regression.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_Toxic_Comments/blob/master/notebooks/a03_text_data_modelling_logistic_regression.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_Toxic_Comments/blob/master/notebooks/a03_text_data_modelling_logistic_regression.ipynb)  |  Losgistic Regression | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| a04_text_data_modelling_spacy.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_Toxic_Comments/blob/master/notebooks/a04_text_data_modelling_spacy.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_Toxic_Comments/blob/master/notebooks/a04_text_data_modelling_spacy.ipynb)  | Topic Modelling using Spacy  | [Bhishan Poudel](https://bhishanpdl.github.io/)  |

# Text Data Processing
For the text data series we can create some features based on the given text. Some feature engineerings are:
```python
Number: letters, capitals, punctuations, symbols, words, sentences, unique words, smileys, qn marks, excl marks
Mean: capitals, word legth
Ratio: num of words / num of unique
```

Basic steps of text processing:
```
Remove: digits, punctuations
Conversion: lowercase
Split: split sentences into words
Stopwords: remove stopwords
Lemmatize: convert word to its base form

```

# Visualization
After doing the preprocessing of the data, we can get more insights into data using some visualization.
![](images/class_distribution.png)
![](images/insult_freq_dist.png)
![](images/toxic_wordcloud.png)
![](images/toxic_tf_idf.png)

# Modelling
For the text classification I used Logistic Regression with following pipelines:
```
preprocess the data and add features
lemmatization
tf-idf for words
tf-idf for characters
then, logistic regression with grid search parameters
```
After searching for hyper parameters I got following results:
```
Accuracy:  0.9516096780643871
Precision:  0.9154411764705882
Recall:  0.532051282051282
F1-score:  0.672972972972973
```

# Model Evaluation
The ROC AUC curve is given below
![](images/roc_auc.png)

# Model Explanation using lime
For the model explanation we can use lime module. For example for one sample here the model predicts the comment to
be non-toxic. Why the model thinks this particular row is classified as non-toxic? We can look the image below:
![](images/lime_example.png)

