# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and load the employee dataset using pandas.read_csv().

2.Convert categorical data into numerical form using get_dummies() and separate features and target variables.

3.Split the dataset into training and testing sets using train_test_split().

4.Create and train the DecisionTreeClassifier model using the training data.

5.Predict the test data results, calculate accuracy, and visualize the decision tree using plot_tree().

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Mithun Kumar V
RegisterNumber:  212225040236
*/

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
data = pd.read_csv("Employee.csv")
data = pd.get_dummies(data, drop_first=True)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
plt.figure(figsize=(20,10))

plot_tree(
    model,
    feature_names=X.columns,
    filled=True
)

plt.show()
```

## Output:
<img width="1501" height="787" alt="image" src="https://github.com/user-attachments/assets/6b0f38f1-ac60-434b-9beb-483009bacc85" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
