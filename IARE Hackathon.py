#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[2]:


df = pd.read_csv('car_evaluation__new.csv')
df


# In[3]:


df.info()


# In[4]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=True,cmap='rainbow')


# In[5]:


df.tail(10)


# In[6]:


df.shape


# In[7]:


print("no.of rows",df.shape[0])


# In[8]:


print("no.of columns",df.shape[1])


# In[9]:


print(df.isnull().values.any())


# In[10]:


sns.heatmap(df.isnull())


# In[11]:


dup_data=df.duplicated().any()
print(dup_data)


# In[12]:


df.describe()


# In[13]:


df.describe(include='all')


# In[14]:


#Training the data


# In[15]:


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Sample data 
data = {
    'buying price': ['vhigh', 'high', 'med', 'low'] * 432,
    'maintenance cost': ['vhigh', 'high', 'med', 'low'] * 432,
    'number of doors': ['2', '3', '4', '5more'] * 432,
    'number of persons': ['2', '4', 'more'] * 576,
    'lug_boot': ['small', 'med', 'big'] * 576,
    'safety': ['low', 'med', 'high'] * 576,
    'decision': ['unacc', 'acc', 'good', 'vgood'] * 432
}

df = pd.DataFrame(data)

# first few rows of the dataset
print(df.head())

# Encode categorical features as numbers
label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split dataset into features and target
X = df.drop(columns=['decision'])
y = df['decision']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

print('Classification Report:')
print(classification_report(y_test, y_pred))

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))


# In[ ]:


#Testing the data


# In[16]:


import joblib

# Sample data 
data = {
    'buying price': ['vhigh', 'high', 'med', 'low'] * 432,
    'maintenance cost': ['vhigh', 'high', 'med', 'low'] * 432,
    'number of doors': ['2', '3', '4', '5more'] * 432,
    'number of persons': ['2', '4', 'more'] * 576,
    'lug_boot': ['small', 'med', 'big'] * 576,
    'safety': ['low', 'med', 'high'] * 576,
    'decision': ['unacc', 'acc', 'good', 'vgood'] * 432
}

df = pd.DataFrame(data)

# Encode categorical features as numbers
label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split dataset into features and target
X = df.drop(columns=['decision'])
y = df['decision']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Save the model
joblib.dump(clf, 'decision_tree_model.pkl')

# Save the label encoders
joblib.dump(label_encoders, 'label_encoders.pkl')


# In[ ]:


#Predicting the data


# In[17]:


# Load the saved model
clf = joblib.load('decision_tree_model.pkl')

# Load the label encoders
label_encoders = joblib.load('label_encoders.pkl')

# Define new data for prediction (ensure it matches the structure of the training data)
new_data = {
    'buying price': ['low', 'med'],
    'maintenance cost': ['low', 'med'],
    'number of doors': ['2', '4'],
    'number of persons': ['2', '4'],
    'lug_boot': ['small', 'big'],
    'safety': ['high', 'med']
}
new_df = pd.DataFrame(new_data)

# Encode the new data using the same label encoders
for column in new_df.columns:
    le = label_encoders[column]
    new_df[column] = le.transform(new_df[column])

# Make predictions
predictions = clf.predict(new_df)

# Optionally decode the predictions back to original labels
decoded_predictions = label_encoders['decision'].inverse_transform(predictions)

# Output the predictions
print("Predicted classes:", decoded_predictions)


# In[ ]:


#Conclusion


# In[ ]:




