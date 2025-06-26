#This is House Rent prediction Webapp using machine learning Linear Regression 
#This Webapp is used to take two inputs as required room or required room area
# it will predict the custom output (Rent price) as we entered while loading custom data 
import streamlit as st        #<-- this is webapp design modoule 
import pandas as pd           
import numpy as np
from sklearn.linear_model import LinearRegression        
from io import StringIO     #<-- this is to store data in the code 
#title of Webapp 
st.title("🏠 House Rent Price Prediction")
#custom data
csv = """house,room,area,price
A,2,50,3000
B,3,100,5000
C,4,150,7000
D,5,200,9000
E,6,250,11000"""
#this loads the data 
housing = pd.read_csv(StringIO(csv))
st.write(" training Data:", housing.head())
# Extract input and target
X = housing[['room', 'area']]
y = housing['price']
# load Linear regression to train 
app = LinearRegression()
#used to train data 
app.fit(X, y)
# Take user input (rrom , area of room)
room = st.number_input("Enter number of rooms:", min_value=1, step=1)
area = st.number_input("Enter area in sqft:", min_value=30, step=10)
# This will Predict and show result as price 
if st.button("Predict Rent Price"):
    input_data = np.array([[room, area]])
    predicted_price = app.predict(input_data)[0]
    st.success(f"💰 Estimated Price: ₹{predicted_price:,.0f}")
