# loding the data
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert to DataFrame for clarity
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y
df.head()

print(y)

print(X)

# checking the data shape
df.shape

# checking the columns
df.columns

# checking the duplicates in the data
df.duplicated().sum()

# this is the duplicated value
df[df.duplicated()]

# Removing the duplicates in the data
df.drop_duplicates(inplace=True)

# checking the null values
df.isnull().sum()

# Stastical information
df.describe()

# information of the data
df.info()

#finding the corrleation
df.corr()

# importing the needed libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# splitting the data
xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=42)

#loding the model
model=LogisticRegression()

#Training the model
model.fit(xtrain,ytrain)

# predecting the data
ypred=model.predict(xtest)

print(ypred)

#evulating the model
accuracy=accuracy_score(ytest,ypred)
print(accuracy)

