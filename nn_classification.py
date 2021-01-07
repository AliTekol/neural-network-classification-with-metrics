#Importing required libraries
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler

#Reading csv files and storing them into seperate variables
data = pd.read_csv("/data.csv")
labels = pd.read_csv("/labels.csv")

#Renaming the empty column to represent sample values
data.columns.values[0] = 'Sample'

#Dropping column that it also present in labels.csv so we can easily work on rest of the data
feature_space = data.drop('Sample', axis = 1)

#We represent the disease types as groups
feature_class = labels['disease_type']

#%25 of the data is set to test group, %75 of the data is set to the training group
training_set, test_set, class_set, test_class_set = train_test_split(feature_space, feature_class, test_size = 0.25, random_state = 42)
class_set = class_set.values.ravel() 
test_class_set = test_class_set.values.ravel()

#Printing model shapes
print("MODEL SHAPES")
print('Training Features Shape:', training_set.shape)
print('Training Labels Shape:', class_set.shape)
print('Testing Features Shape:', test_set.shape)
print('Testing Labels Shape:', test_class_set.shape)
#OUTPUT: 
#Training Features Shape: (266, 1836)
#Training Labels Shape: (266,)
#Testing Features Shape: (89, 1836)
#Testing Labels Shape: (89,)

#Multi-layer Perceptron is sensitive to feature scaling, so we will normalise the data to get meaningful results
scaler = StandardScaler()

#Fit only to the training data
scaler.fit(training_set)

#Applying the transformation to the training data:
training_set = scaler.transform(training_set)

#Applying same transformation to test data
test_set = scaler.transform(test_set)

#We will show 5 different outputs by changing MLPClassifier parameters and we will define the variables such as mlp_1 mlp_2 mlp_3 and mlp_4
#In this variable, we set ReLu activation function and other hyperparameters below
mlp_1 = MLPClassifier(activation = 'relu', hidden_layer_sizes = (30, 30, 30), max_iter = 100, random_state = None, solver = 'lbfgs')

#Now that the model has been made we can fit the training data to our model, 
# remember that this data has already been processed and scaled:
mlp_1.fit(training_set, class_set)

#To get predictions, we simply call the predict() method of our fitted model:
predictions = mlp_1.predict(test_set)

#Calculating Precision, Recall and F2 measure formulas
precision = precision_score(test_class_set, predictions, average = None)
print("----------------------------------------")
print("PERFORMANCE RESULTS ON FIRST VARIABLE (MLP_1)")
print("PRECISION:", precision)

recall = recall_score(test_class_set, predictions, average = None)

print("RECALL:", recall)

F2_measure = 5 * precision * recall / (4 * precision + recall)

print("F2 Measure:", F2_measure)

#In this variable, we set the logistic sigmoid function as activation funciton and changed other hyperparameters
mlp_2 = MLPClassifier(activation = 'logistic', hidden_layer_sizes = (150, 100, 50), max_iter = 200, random_state = 1, solver = 'adam')

mlp_2.fit(training_set, class_set)

predictions = mlp_2.predict(test_set)

precision = precision_score(test_class_set, predictions, average = None)
print("----------------------------------------")
print("PERFORMANCE RESULTS ON SECOND VARIABLE (MLP_2)")
print("PRECISION:", precision)

recall = recall_score(test_class_set, predictions, average = None)

print("RECALL:", recall)

F2_measure = 5 * precision * recall / (4 * precision + recall)

print("F2 Measure:", F2_measure)

#In this variable, we set the identity activation funciton and changed other hyperparameters
mlp_3 = MLPClassifier(activation = 'identity', hidden_layer_sizes = (10, 30, 20), max_iter = 300, random_state = 2, solver = 'sgd')

mlp_3.fit(training_set, class_set)

predictions = mlp_3.predict(test_set)

precision = precision_score(test_class_set, predictions, average = None)
print("----------------------------------------")
print("PERFORMANCE RESULTS ON THIRD VARIABLE (MLP_3)")
print("PRECISION:", precision)

recall = recall_score(test_class_set, predictions, average = None)

print("RECALL:", recall)

F2_measure = 5 * precision * recall / (4 * precision + recall)

print("F2 Measure:", F2_measure)

#In this variable, we set the tanh activation funciton and changed other hyperparameters
mlp_4 = MLPClassifier(activation = 'tanh', hidden_layer_sizes = (50, 50, 50), max_iter = 400, random_state = 3, solver = 'lbfgs')

mlp_4.fit(training_set, class_set)

predictions = mlp_4.predict(test_set)

precision = precision_score(test_class_set, predictions, average = None)
print("----------------------------------------")
print("PERFORMANCE RESULTS ON FOURTH VARIABLE (MLP_4)")
print("PRECISION:", precision)

recall = recall_score(test_class_set, predictions, average = None)

print("RECALL:", recall)

F2_measure = 5 * precision * recall / (4 * precision + recall)

print("F2 Measure:", F2_measure)
print("----------------------------------------")
"""
We get metrics like METRIC: [first_class_metric_value   second_class_metric_value    third_class_metric_value    fourth_class_metric_value]

CODE OUTPUT:
----------------------------------------
PERFORMANCE RESULTS ON FIRST VARIABLE (MLP_1)
PRECISION: [1.         0.96774194 0.42857143 0.87096774]
RECALL: [0.8        0.96774194 1.         0.9       ]
F2 Measure: [0.83333333 0.96774194 0.78947368 0.89403974]
----------------------------------------

From the above values, we can get a classified table like below.

                        PRECISION    RECALL       F2 MEASURE

    breast cancer       1.00         0.8          0.83333333
    colon cancer        0.96774194   0.96774194   0.96774194
    lung cancer         0.42857143   1.00         0.78947368 
    prosrtate cancer    0.87096774   0.9          0.89403974

----------------------------------------
PERFORMANCE RESULTS ON FIRST VARIABLE (MLP_2)
PRECISION: [0.9047619  0.93939394 0.25       0.87096774]
RECALL: [0.76       1.         0.33333333 0.9       ]
F2 Measure: [0.78512397 0.98726115 0.3125     0.89403974]
----------------------------------------

From the above values, we can get a classified table like below.

                        PRECISION    RECALL       F2 MEASURE

    breast cancer       0.9047619    0.76          0.78512397
    colon cancer        0.93939394   1.00          0.98726115
    lung cancer         0.25         0.33333333    0.3125
    prosrtate cancer    0.87096774   0.9           0.89403974

----------------------------------------
PERFORMANCE RESULTS ON FIRST VARIABLE (MLP_3)
PRECISION: [1.         0.93548387 0.75       0.83333333]
RECALL: [0.72       0.93548387 1.         1.        ]
F2 Measure: [0.76271186 0.93548387 0.9375     0.96153846]
----------------------------------------

From the above values, we can get a classified table like below.

                        PRECISION    RECALL       F2 MEASURE

    breast cancer       1.00         0.72         0.76271186
    colon cancer        0.93548387   0.93548387   0.93548387
    lung cancer         0.75         1.00         0.9375
    prosrtate cancer    0.83333333   1.00         0.96153846
    
----------------------------------------
PERFORMANCE RESULTS ON FIRST VARIABLE (MLP_4)
PRECISION: [0.95       0.91176471 0.5        0.90322581]
RECALL: [0.76       1.         0.66666667 0.93333333]
F2 Measure: [0.79166667 0.98101266 0.625      0.92715232]
----------------------------------------

From the above values, we can get a classified table like below.

                        PRECISION    RECALL       F2 MEASURE

    breast cancer       0.95         0.76         0.79166667
    colon cancer        0.91176471   1.00         0.98101266
    lung cancer         0.5          0.66666667   0.625
    prosrtate cancer    0.90322581   0.93333333   0.92715232]
    
----------------------------------------
"""