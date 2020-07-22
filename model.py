import pandas as pd
data=pd.read_csv(r'E:\Python,ML,DL,NLP\Completed\Brest Cancer data\datasets_180_408_data.csv')

import seaborn as sns
ax = sns.countplot(data['diagnosis'], label= 'Count')
B,M = data['diagnosis'].value_counts()
#print('Benign', B)
#print('Malignanat', M)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

del data['Unnamed: 32']

X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train=pd.DataFrame(X_train)

X_test=pd.DataFrame(X_test)

from sklearn.ensemble import RandomForestClassifier

rf =RandomForestClassifier()
rf.fit(X_train,y_train)

import pickle
pickle.dump(rf,open('model.pkl','wb'))