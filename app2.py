
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Chennai House Prediction App")


with open('reg_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


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

selected_columns_location = data.iloc[:,6:68]
location_names = [column[9:] for column in selected_columns_location.columns]

selected_columns_builder = data.iloc[:,68:125]
builder_names = [column[8:] for column in selected_columns_builder.columns]

# Extract unique labels from 'builder' and 'location' columns
builder_options = np.concatenate([['Select Builder'], builder_names])
location_options = np.concatenate([['Select Location'],location_names])

area = st.number_input(r"$\textsf{\Large Area}$",min_value=0.0, value=1000.0, step=1.0)
status = st.selectbox(r"$\textsf{\Large Status (under constuction -> 0 | ready to move -> 1)}$",options=[0,1])
bhk = st.number_input(r"$\textsf{\Large Number of Bedrooms}$", min_value=1, value=2, step=1)
bathroom = st.number_input(r"$\textsf{\Large Number of Bathrooms}$", min_value=0.0, value=1.0, step=0.5)
age = st.number_input(r"$\textsf{\Large Age}$", min_value=0.0, value=5.0, step=1.0)
location = st.selectbox(r"$\textsf{\Large Location}$", location_options)
builder = st.selectbox(r"$\textsf{\Large Builder}$", builder_options)

user_input = pd.DataFrame({
    'area': [area],
    'bhk': [bhk],
    'bathroom': [bathroom],
    'age': [age],
    'status_Under Construction' : [status],
})
for column in data.iloc[:,6:68].columns:
    if column[9:] == location:
        user_input[column] = 1.0
    else:
        user_input[column] = 0.0

for column in data.iloc[:,68:125].columns:
    if column[8:] == builder:
        user_input[column] = 1.0
    else:
        user_input[column] = 0.0

if st.button("Generate Prediction"):

    prediction = loaded_model.predict(user_input)
    rounded_prediction = round(prediction[0], 2)
    # Display the prediction
    st.markdown(f"<h1 style='text-align: center;'>Predicted House Price: {rounded_prediction} lakhs (INR)</h1>", unsafe_allow_html=True)
