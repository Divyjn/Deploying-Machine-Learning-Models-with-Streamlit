from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
import pickle

x, y = load_wine(return_X_y=True)
mdl = RandomForestClassifier().fit(x, y)
with open('wine_model.pkl', 'wb') as f:
    pickle.dump(mdl, f) 