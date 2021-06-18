#importing librery

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#making dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x= dataset.iloc[:,0:-1].values
y = dataset.iloc[:,-1].values


#spliting dataset into training and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


#featuring data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#training model
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train,y_train)


#making prediction
y_pred = classifier.predict(x_test)
y_pred = y_pred.reshape(len(y_pred),1)
#importing confusion matrix and accuracy_score

from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test,y_pred))