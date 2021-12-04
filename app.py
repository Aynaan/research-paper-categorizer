from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from model import Preprocessing
import pickle
import pandas as pd
from tensorflow import keras
# Creating FastAPI instance
app = FastAPI()
 
# Creating class to define the request body
# and the type hints of each attribute
class request_body(BaseModel):
    title=str()
    abstract=str()



@app.post('/predict')
def predict(data : request_body):
    title=data.title
    abstract=data.abstract
    pp=Preprocessing()
    title_input = pd.Series(title)
    title_cleaned, title_vectorized = pp.preprocess(title_input, training="title")

    abstract_input = pd.Series(abstract)
    abstract_cleaned, abstract_vectorized = pp.preprocess(abstract_input, training="abstract")

    title_vectorized=title_vectorized.toarray()
    abstract_vectorized=abstract_vectorized.toarray()

    model = keras.models.load_model("my_model.h5",compile=False)
    y_probs=model.predict({"title": title_vectorized, "abstract": abstract_vectorized})
    y_dict={"ComputerScience":y_probs[0][:].flatten(),"Physics":y_probs[1][:].flatten(),'Mathematics':y_probs[2][:].flatten(),'Statistics':y_probs[3][:].flatten(),'QuantitativeBiology':y_probs[4][:].flatten(),'QuantitativeFinance':y_probs[5][:].flatten()}
    categories=['ComputerScience', 'Physics', 'Mathematics', 'Statistics',  'QuantitativeBiology', 'QuantitativeFinance']
    threshold=[0.4666402041912079,0.41269561648368835,0.41028860211372375,0.40249261260032654,0.21160995960235596,0.1636740267276764]
    prediction=[]
    for (i,j) in zip(categories,threshold):
       temp= y_dict[i][0]
       if(temp>=j):
           prediction.append(i)

    return prediction
   
