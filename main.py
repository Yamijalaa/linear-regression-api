from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

#Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

#Input Schema
class InputData(BaseModel):
    experience: float

@app.post("/predict")
def predict(data:InputData):
    x = np.array([[data.experience]])
    prediction = model.predict(x)[0]
    return {"predicted_salary":prediction}
