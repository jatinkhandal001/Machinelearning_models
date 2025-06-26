import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from io import StringIO
# Title
st.title("🏠 House Rent Price Prediction")
csv = """house,room,area,price
A,2,50,3000
B,3,100,5000
C,4,150,7000
D,5,200,9000
E,6,250,11000"""



housing = pd.read_csv(StringIO(csv))
#st.write(" training Data:", housing.head())






# Extract features and target
X = housing[['room', 'area']]
y = housing['price']

# Train the model
app = LinearRegression()
app.fit(X, y)

# Take user input
room_input = st.number_input("Enter number of rooms:", min_value=1, step=1)
area_input = st.number_input("Enter area in sqft:", min_value=30, step=10)

# Predict and show result
if st.button("Predict Price"):
    input_data = np.array([[room_input, area_input]])
    predicted_price = app.predict(input_data)[0]
    st.success(f"💰 Estimated Price: ₹{predicted_price:,.0f}")
