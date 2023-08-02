#!/usr/bin/env python
# coding: utf-8

# <h3>1.Task 1: Dataset Selection 
# A.Choose two datasets from the provided repositories. 
# Dataset 1: Titanic Dataset
# Dataset 2: FIFA-21 Complete Dataset</h3>

# <h3>B.Justify your selection for each dataset based on its relevance to machine learning tasks.
# Include a brief paragraph explaining the dataset's potential for analysis and its suitability for machine learning applications.
# Dataset 1: Titanic Dataset
# Justification:The Titanic dataset is a renowned and long-lasting dataset in the disciplines of data science and machine learning.
# It contains information about the passengers that boarded the RMS Titanic on its tragic first voyage,
# including whether or not they survived. The dataset may be used for machine learning applications
# since it contains categorical and numerical data that can be trained to 
# create a machine learning model that can predict whether or not a passenger survived the accident based on their features.
# 
# 
# Dataset 2: FIFA-21 Complete Dataset
# Justification: The FIFA-21 Complete Dataset encompasses player ratings, ages, nationalities, positions, and potential for
# the future for machine learnig this dataset provides valuable features for analysis and modeling, but care should be 
# taken to account for potential discrepancies caused by transfer updates.</h3>

# <h3>Task 2: Data Exploration with Python 
# A.Perform exploratory data analysis (EDA) using Python for the first dataset. 
# B.Generate summary statistics, identify data types, and visualize the data distribution to gain insights into the dataset.</h3>

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[2]:


df = pd.read_csv('E:/Lab 10/titanic.csv', encoding = 'ISO-8859-1')


# In[3]:


print("Preview of the dataset:")
print(df.head())


# In[4]:


print("\nDataset Information")
print(df.info())


# In[5]:


print("\nSummary Statistics")
print(df.describe())


# In[6]:


print("\nHistograms:")
df.hist(figsize=(10,8))
plt.title("Histogram")
plt.tight_layout()
plt.show()


# In[7]:


sns.boxplot(x='Age', data=df)
plt.xlabel('Survived')
plt.ylabel('Age')
plt.title('Age Distribution based on Survival')
plt.show()


# <h3>Task 3: Data Preprocessing with Python 
# a.Preprocess the data from the first dataset using Python. 
# b.Handle missing values, outliers, and perform feature engineering when necessary to prepare the data for machine learning models.</h3>

# In[8]:


print("\nMissing values:")
print(df.isnull().sum())


# In[9]:


# Removing outliers
df = df[df['Age'] < 60]  # Remove outliers where age is greater than 60


# In[10]:


# Feature engineering
df['age_squared'] = df['Age'] ** 2  # Add new feature: age_squared


# In[11]:


df.drop(['Cabin','Name','Ticket','PassengerId'], axis =1, inplace = True) 
print(df.head())


# In[14]:


#Converting 'Sex' and 'Embarked' into category
df['Sex']=pd.factorize(df['Sex'])[0]
df['Embarked']=pd.factorize(df['Embarked'])[0]


# <h3>Task 4: Implement Machine Learning Models with Python 
# Implement at least two different machine learning models (e.g., SVM, Random Forest, Neural Network) for the first dataset using Python. 
# Evaluate and compare the performance of each model using appropriate metrics to determine the most suitable model for the dataset.
# </h3>

# In[16]:


X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[25]:


svm_model = SVC(kernel='linear', random_state=1)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of SVM: {svm_accuracy:.2f}')
classification_rep = classification_report(y_test, y_pred)
print('Classification Report:\n', classification_rep)


# In[19]:


random_forest_model = RandomForestClassifier(random_state=1)
random_forest_model.fit(X_train, y_train)

y_pred = random_forest_model.predict(X_test)
frst_accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of Random Forest: {frst_accuracy:.2f}')



classification_rep = classification_report(y_test, y_pred)
print('Classification Report:\n', classification_rep)


# Looking at both results Random Forest it is a close call between both models but  Forest Model is slightly better than SVM.

# <h3>Task 5: Visualization with Python 
# A.Create meaningful visualizations (e.g., scatter plots, heatmaps, bar charts) for the first dataset using Python. 
# B.Use libraries like Matplotlib, Seaborn, or Plotly to create clear and insightful visual representations of the dataset.</h3>

# In[20]:


#heat map
correlation_matrix = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot = True, cmap="coolwarm")
plt.show()


# In[21]:


#Box plot Survival Count by Pclass 
plt.figure(figsize=(8, 6))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival Count by Pclass')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()


# In[22]:


#Fare Distribution by Pclass
plt.figure(figsize=(10, 6))
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title('Fare Distribution by Pclass')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.show()


# In[23]:


#Survival Count by Embarked Port
plt.figure(figsize=(8, 6))
sns.countplot(x='Embarked', hue='Survived', data=df)
plt.title('Survival Count by Embarked Port')
plt.xlabel('Embarked Port')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()


# In[24]:


#Scatter Plot of Age vs. Fare by Survival
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
plt.title('Scatter Plot of Age vs. Fare by Survival')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

