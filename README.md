# Diabetes Prediction Using Machine Learning

This is a diabetes prediction model using machine learning. The model is built using the Support Vector Machine (SVM) algorithm and utilizes a dataset containing various health-related features to predict the likelihood of an individual having diabetes.


## Acknowledgements

 - [National Library of Medicine](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4418458/)

 
## Prerequisites

Before running the code, ensure that you have the necessary libraries installed. You can install them using the following commands:

```bash
 pip install pandas numpy matplotlib seaborn scikit-learn
```
## About the Dataset

### Dataset Source:

  https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

### Dataset Description

This ML model to predict diabetes utilizes Pima Indian Diabetes dataset, which focuses on the health of Pima Indians—a Native American community residing in Mexico and Arizona, USA. This particular group has been identified with a notable prevalence of diabetes mellitus, making research on this population crucial and reflective of broader global health concerns. The dataset specifically includes health information from Pima Indian females aged 21 years and older, and it serves as a widely used benchmark dataset. Given its relevance, this dataset is particularly meaningful for studying health patterns within underrepresented minority or indigenous groups.


## Code Structure

### Importing Libraries
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.preprocessing import StandardScaler


# Importing Libraries


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.preprocessing import StandardScaler

# Loading the Dataset


data = pd.read_csv('diabetes_data.csv')

# Checking for Missing Values

sns.heatmap(data.isnull())

# Exploring Corelation

correlation = data.corr()

sns.heatmap(correlation)

# Splitting the Dataset - Data vs Label

X = data.drop('Outcome', axis=1)
Y = data['Outcome']

# Standardizing the Data

scaler = StandardScaler()
scaler.fit(X)
standard_data = scaler.transform(X)
X = standard_data

# Splitting the Data into Train vs Test

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Training the Model

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Evaluation of the Model

prediction_X_train = classifier.predict(X_train)
accuracy_train_data = accuracy_score(prediction_X_train, Y_train)

prediction_X_test = classifier.predict(X_test)
accuracy_test_data = accuracy_score(prediction_X_test, Y_test)

# Making a Predictive System

input_data = (6, 148, 72, 35, 0, 33.6, 0.627, 50)
input_data_as_array = np.asarray(input_data)
input_data_reshaped = input_data_as_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)
prediction = classifier.predict(std_data)

# Saving and Loading the Model

import pickle
filename = 'diabetes_model.sav'
pickle.dump(classifier, open(filename, 'wb'))
loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))

# Results

After training the model and evaluating its performance on both training and testing datasets, the accuracy scores are printed. Additionally, a predictive system is implemented, allowing users to input their health data and receive a prediction regarding diabetes risk.
