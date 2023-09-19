#!/usr/bin/env python
# coding: utf-8

# In[55]:


## import necessary library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
## Read the dataset, convert it into dataframe
# columns_of_interest = ['id','title', 'text', 'label']
dataframe = pd.read_csv(r'C:\Users\NANDESHVAR\Downloads\Fake\train.csv')
dataframe.head()


# In[56]:


dataframe.shape


# In[57]:


dataframe.columns


# In[58]:


dataframe.isnull().sum()


# In[59]:


dataframe = dataframe.dropna()


# In[13]:


dataframe.isnull().sum()


# In[60]:


# merging the author name and news title
dataframe['content'] = dataframe['author']+' '+dataframe['title']


# In[61]:


print(dataframe['content'])


# In[62]:


# separating the data & label
X = dataframe.drop(columns='label', axis=1)
Y = dataframe['label']


# In[63]:


print(X)
print(Y)


# In[64]:


import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
port_stem = PorterStemmer()


# In[65]:


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# In[66]:


dataframe['content'] = dataframe['content'].apply(stemming)


# In[67]:


print(dataframe['content'])


# In[68]:


#separating the data and label
X = dataframe['content'].values
Y = dataframe['label'].values


# In[69]:


print(X)


# In[70]:


print(Y)


# In[71]:


Y.shape


# In[72]:


vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)


# In[73]:


print(X)


# In[74]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)


# In[75]:


model = LogisticRegression()


# In[76]:


model.fit(X_train, Y_train)


# In[77]:


# accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[78]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[79]:


# accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy_logistic = accuracy_score(X_test_prediction, Y_test)


# In[80]:


print('Accuracy score of the test data : ', test_data_accuracy_logistic)


# In[81]:


X_new = X_test[3]

prediction = model.predict(X_new)
print(prediction)

if (prediction[0]==3):
  print('The news is Real')
else:
  print('The news is Fake')


# In[82]:


print(Y_test[5])


# In[83]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create a Decision Tree Classifier
model = DecisionTreeClassifier(random_state=2)

# Train the model
model.fit(X_train, Y_train)

# Make predictions on the training data
X_train_prediction = model.predict(X_train)

# Calculate training data accuracy
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score on the training data:', training_data_accuracy)


# In[84]:


X_test_prediction = model.predict(X_test)

# Calculate test data accuracy
test_data_accuracy_decision_tree = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on the test data:', test_data_accuracy)


# In[85]:


# Accuracy scores
accuracy_scores = [training_data_accuracy, test_data_accuracy_decision_tree, test_data_accuracy_logistic]

# Algorithm labels
algorithms = ['Training Data', 'Decision Tree', 'Logistic Regression']

# Create a bar chart
plt.figure(figsize=(8, 5))
plt.bar(algorithms, accuracy_scores, color=['blue', 'green', 'orange'])
plt.xlabel('Algorithm')
plt.ylabel('Accuracy Score')
plt.title('Comparison of Algorithm Accuracy')
plt.ylim(0.95, 1.0)  # Set y-axis limits for better visualization
plt.tight_layout()

# Display the chart
plt.show()


# In[86]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Predict on training data
Y_train_pred = model.predict(X_train)
# Predict on testing data
Y_test_pred = model.predict(X_test)

# Calculate accuracy
training_accuracy = accuracy_score(Y_train, Y_train_pred)
testing_accuracy = accuracy_score(Y_test, Y_test_pred)

# Calculate precision
training_precision = precision_score(Y_train, Y_train_pred)
testing_precision = precision_score(Y_test, Y_test_pred)

# Calculate recall
training_recall = recall_score(Y_train, Y_train_pred)
testing_recall = recall_score(Y_test, Y_test_pred)

# Calculate F1-score
training_f1 = f1_score(Y_train, Y_train_pred)
testing_f1 = f1_score(Y_test, Y_test_pred)

# Print the evaluation metrics
print("Training Accuracy:", training_accuracy)
print("Testing Accuracy:", testing_accuracy)
print("Training Precision:", training_precision)
print("Testing Precision:", testing_precision)
print("Training Recall:", training_recall)
print("Testing Recall:", testing_recall)
print("Training F1-Score:", training_f1)
print("Testing F1-Score:", testing_f1)


# In[ ]:




