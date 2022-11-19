import numpy as np
import pandas as pd
import warnings
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")

# load training data
data = np.array(pd.read_csv("sample_data.csv"))
x = data[1:, 1:-1].astype('int')
y = data[1:, -1].astype('int')

# preparing the model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
mod = LogisticRegression()
mod.fit(x_train, y_train)
weights = [np.array([50, 30, 60])]

# train the model
mod.predict_proba(weights)

# dump the model
pickle.dump(mod, open('my_model.model', 'wb'))


