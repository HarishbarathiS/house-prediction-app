from HousePrices import HousePrice
import uvicorn
from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

file_path = "clean_data.csv"  
df = pd.read_csv(file_path)

data = df

builder_data = data['builder'].value_counts()
remove_builder = builder_data[builder_data<=10]
data['builder'] = data['builder'].apply(lambda x: 'other' if x in remove_builder else x)


location_data = data['location'].value_counts()
remove_location = location_data[location_data<10]
data['location'] = data['location'].apply(lambda x: 'other' if x in remove_location else x)


data = pd.get_dummies(df, drop_first = True)

pickle_load = open('reg_model.pkl', 'rb')
model = pickle.load(pickle_load)

@app.get('/')
def index():
    return {'message': 'Hello world'}

@app.get('/{name}')
def get_name(name: str):
    return {f'Welcome to {name}\'s galaxy': f'{name}'}

@app.post('/predict')
def predict(val:HousePrice):
    val = val.__dict__
    user_input = pd.DataFrame({
    'area':[val["area"]],
    'bhk': [val["bhk"]],
    'bathroom': [val["bathroom"]],
    'age': [val["age"]],
    'status_Under Construction' : [val["status"]],
    })

    for column in data.iloc[:,6:68].columns:
        if column[9:] == val["location"]:
            user_input[column] = [1.0]
        else:
            user_input[column] = [0.0]

    for column in data.iloc[:,68:125].columns:
        if column[8:] == val["builder"]:
            user_input[column] = [1.0]
        else:
            user_input[column] = [0.0]
    prediction = model.predict(user_input)
    return {'Predicted House Price' : f"{prediction}"}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, reload=True)

