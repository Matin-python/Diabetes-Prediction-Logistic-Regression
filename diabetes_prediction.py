import numpy as np
import pandas as pd
import seaborn as sns

import sklearn.metrics as sm
from sklearn import linear_model
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


df = pd.read_csv('diabetes2.csv')

sns.countplot (x= 'Outcome', data= df)

plt.subplots(figsize = (9, 9))
sns.heatmap(df.corr(), annot= True)
plt.show()

x = df.drop(["Outcome"], axis=1) 
y = df.Outcome

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

reg_logestic = linear_model.LogisticRegression()
reg_logestic.fit(x_train, y_train)

out_pred = reg_logestic.predict(x_test)

err = np.abs(y_test - out_pred) 
correct_percentage = (1 - err.mean()) * 100
print("-"*50)
print("correct prediction= ", correct_percentage, "%")

msr = sm.mean_squared_error(y_test, out_pred)
print('mean squared error= ', msr)
print("-"*50)