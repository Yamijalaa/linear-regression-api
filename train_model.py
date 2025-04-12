from sklearn.linear_model import LinearRegression
import pickle
import numpy as np

#Training Data
X = np.array([[1],[2],[3],[4],[5]]) #years of experience
y = np.array([30000,35000,40000,45000,50000]) #salary

#Train Model
model = LinearRegression()
model.fit(X,y)

#Save the model
with open("model.pkl","wb") as f:
    pickle.dump(model,f)