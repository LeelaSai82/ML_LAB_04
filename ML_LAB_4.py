#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import math

# Create a DataFrame with the given data
data = {
    'age': ['<=30', '<=30', '31...40', '>40', '>40', '>40', '31...40', '<=30', '<=30', '>40', '<=30', '31...40', '31...40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'high'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes']
}

df = pd.DataFrame(data)

# Calculate the entropy of a target variable
def entropy(target):
    counts = target.value_counts()
    entropy = 0
    total = len(target)
    for count in counts:
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

# Calculate the information gain for a feature
def information_gain(data, feature, target):
    unique_values = data[feature].unique()
    weighted_entropy = 0
    for value in unique_values:
        subset = data[data[feature] == value]
        weighted_entropy += len(subset) / len(data) * entropy(subset[target])
    
    return entropy(data[target]) - weighted_entropy

# Target variable
target = 'buys_computer'

# Calculate information gain for each feature
features = ['age', 'income', 'student', 'credit_rating']
information_gains = {feature: information_gain(df, feature, target) for feature in features}

# Find the feature with the highest information gain
root_node = max(information_gains, key=information_gains.get)

print("Information Gains:")
for feature, gain in information_gains.items():
    print(f"{feature}: {gain}")
    

print(f"The root node for the decision tree is: {root_node}")


# In[4]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Create a DataFrame with the given data
data = {
    'age': ['<=30', '<=30', '31...40', '>40', '>40', '>40', '31...40', '<=30', '<=30', '>40', '<=30', '31...40', '31...40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'high'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes']
}

df = pd.DataFrame(data)

# Encode categorical features to numerical values
label_encoder = LabelEncoder()
for col in df.columns:
    if col != 'buys_computer':
        df[col] = label_encoder.fit_transform(df[col])

# Split the data into features (X) and target variable (y)
X = df.drop('buys_computer', axis=1)
y = df['buys_computer']

# Create and fit the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X, y)

# Get the depth of the constructed tree
tree_depth = model.tree_.max_depth
print(f"The depth of the constructed Decision Tree is: {tree_depth}")


# In[5]:


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))  # You can adjust the figsize as needed
plot_tree(model, filled=True, feature_names=data['age'], class_names=['age', 'student'], rounded=True)
plt.show()


# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#loading the project data
df = pd.read_excel(r"C:\Users\saite\Downloads\embeddingsdata (1).xlsx")
df


# In[7]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Create a Decision Tree classifier
model = DecisionTreeClassifier()
binary_df = df[df['Label'].isin([0, 1])]
X = binary_df[['embed_1', 'embed_2']]
y = binary_df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Fit the model on the training data
model.fit(X_train, y_train)

# Calculate training set accuracy
train_accuracy = model.score(X_train, y_train)

# Calculate test set accuracy
test_accuracy = model.score(X_test, y_test)

# Print the accuracies
print(f"Training Set Accuracy: {train_accuracy}")
print(f"Test Set Accuracy: {test_accuracy}")
print("hello")

# In[ ]:

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Create a Decision Tree classifier with max_depth constraint
model = DecisionTreeClassifier(max_depth=5)

# Fit the model on the training data
binary_df = df[df['Label'].isin([0, 1])]
X = binary_df[['embed_1', 'embed_2']]
y = binary_df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Fit the model on the training data
model.fit(X_train, y_train)

# Calculate training set accuracy
train_accuracy = model.score(X_train, y_train)

# Calculate test set accuracy
test_accuracy = model.score(X_test, y_test)

# Print the accuracies
print(f"Training Set Accuracy (max_depth=5): {train_accuracy}")
print(f"Test Set Accuracy (max_depth=5): {test_accuracy}")

# Visualize the tree with max_depth constraint
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=['embed_0', 'embed_1'], class_names=['no', 'yes'], rounded=True)
plt.show()



