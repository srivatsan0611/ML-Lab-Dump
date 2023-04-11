import pandas as pd
import numpy as np

data = pd.read_csv("Iris.csv")
data = data.drop("Id",axis=1)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data["Species"] = le.fit_transform(data["Species"])

from sklearn.tree import DecisionTreeClassifier,plot_tree

DT = DecisionTreeClassifier()

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0,test_size=0.3)

DT.fit(X_train,y_train)

import pickle

with open("model.pkl","wb") as f:
    pickle.dump(DT,f)

    #%%
import pickle
model=pickle.load(open("model.pkl","rb"))
# %%
