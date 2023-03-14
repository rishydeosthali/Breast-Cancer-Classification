# Import Python tools for loading/navigating data
import os             # Good for navigating your computer's files 
import numpy as np    # Great for lists (arrays) of numbers
import pandas as pd   # Great for tables (google spreadsheets, microsoft excel, csv)
from sklearn.metrics import accuracy_score   # Great for creating quick ML models
from MLModels import * # Boundary Classifier to determine type of cancer

# Import data visualization tools
import seaborn as sns
import matplotlib.pyplot as plt 

# Use the 'pd.read_csv('file')' function to read in read our data and store it in a variable called 'dataframe'
data_path  = 'cancer.csv'
dataframe = pd.read_csv(data_path)

dataframe = dataframe[['diagnosis', 'perimeter_mean', 'radius_mean', 'texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean', 'symmetry_mean']]
dataframe['diagnosis_cat'] = dataframe['diagnosis'].astype('category').map({'M': '1 (malignant)', 'B': '0 (benign)'})

# ---------------- Boundary classifier testing ---------------
 
chosen_boundary = 10 #Try changing this!
input_parameter = 'radius_mean'

y_pred = boundary_classifier(chosen_boundary, dataframe[input_parameter])
dataframe['predicted'] = y_pred

y_true = dataframe['diagnosis']

sns.catplot(x = 'radius_mean', y = 'diagnosis_cat', hue = 'predicted', data = dataframe, order=['1 (malignant)', '0 (benign)'])
plt.plot([chosen_boundary, chosen_boundary], [-.2, 1.2], 'g', linewidth = 2)
plt.show()

# ---------------- Boundary Classifier Testing -----------------

# --------------- Linear Regression Testing ----------------

X = dataframe[['radius_mean']]
y = dataframe['diagnosis'].astype('category').map({'M': '1', 'B': '0'})

predictions = linreg(X,y)

sns.scatterplot(x='radius_mean', y='diagnosis', data=dataframe)
plt.plot(X, predictions, color='r')
plt.legend(['Linear Regression Fit', 'Data'])
plt.show()

# -------------- Linear Regression Testing ----------

# -------------- Logistic Regression Testing with single variable --------------------

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(dataframe, test_size = 0.2, random_state = 1) # Split input data into training data and test data

# Extract training and test data
X = ['radius_mean']
y = 'diagnosis'
X_test = test_df[X]
y_test = test_df[y].astype('category').map({'M': '1', 'B': '0'})
X_train = train_df[X]
y_train = train_df[y].astype('category').map({'M': '1', 'B': '0'})

# Train model on variable
logreg_model = logreg_train(X_train,y_train)

# Predict diagnosis using trained model
y_pred = logreg_predict(logreg_model, X_test)

test_df['predicted'] = y_pred.squeeze()
sns.catplot(x = X[0], y = 'diagnosis_cat', hue = 'predicted', data=test_df, order=['1 (malignant)', '0 (benign)'])
plt.plot([chosen_boundary, chosen_boundary], [-.2, 1.2], 'g', linewidth = 2)
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# --------------- Logistic Regression Testing with single variable -------------

# -------------- Logistic Regression Testing with multiple variables --------------------

multi_X = ['radius_mean','concavity_mean']
y ='diagnosis'

# 1. Split data into train and test
multi_train_df, multi_test_df = train_test_split(dataframe, test_size = 0.2, random_state = 1)

# 2. Prepare your X_train, X_test, y_train, and y_test variables by extracting the appropriate columns:
X_train = multi_train_df[multi_X]
y_train = multi_train_df[y].astype('category').map({'M': '1', 'B': '0'})
X_test = multi_test_df[multi_X]
y_test = multi_test_df[y].astype('category').map({'M': '1', 'B': '0'})


# 3. Train model
logreg_model = logreg_train(X_train,y_train)

# 4 Predict diagnosis using trained model
y_pred = logreg_predict(logreg_model, X_test)

# 5. Evaluate the accuracy by comparing to to the test labels and print out accuracy.
#test_df['predicted'] = y_pred.squeeze()
#sns.catplot(x = X[0], y = 'diagnosis_cat', hue = 'predicted', data=test_df, order=['1 (malignant)', '0 (benign)'])
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


# -------------- Logistic Regression Testing with multiple variables --------------------

