#%%
import pandas as pd
import numpy as np

#%%

data = pd.read_csv("Iris.csv")
data.head()
# %%

data = data.drop("Id",axis=1)
# %%

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data.iloc[:,-1] = le.fit_transform(data.iloc[:,-1])
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
# %%
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()

DT.fit(X,y)
# %%
import pickle
with open('model.pkl','wb') as f:
    pickle.dump(DT,f)
# %%
