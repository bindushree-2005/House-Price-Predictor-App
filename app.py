import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Title
st.title("🏠 House Price Prediction App")

# Load dataset
data = pd.read_csv("small_data.csv")


# Clean column names
data.columns = data.columns.str.strip()

# Convert numeric columns
for col in ['Carpet Area', 'Bathroom', 'Balcony', 'Price (in rupees)', 'Super Area']:
    data[col] = data[col].astype(str).str.replace('[^0-9.]', '', regex=True)
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop missing values
data = data.dropna(subset=['Carpet Area', 'Bathroom', 'Balcony', 'Price (in rupees)'])

# Encode location (simple method)
data['location'] = data['location'].astype('category').cat.codes

# Select only numeric columns
# Keep only required columns
data = data[['Carpet Area', 'Bathroom', 'Balcony', 'Price (in rupees)', 'location']]

# Drop missing values only for these columns
data = data.dropna()
# Features and target
X = data.drop('Price (in rupees)', axis=1)
y = data['Price (in rupees)']

# Train model
model = LinearRegression()
model.fit(X, y)

# UI Inputs
st.header("Enter House Details")

area = st.number_input("Carpet Area (sqft)", min_value=100)
bath = st.number_input("Number of Bathrooms", min_value=1)
balcony = st.number_input("Number of Balconies", min_value=0)
location = st.number_input("Location Code (0-10 approx)", min_value=0)

# Predict button
if st.button("Predict Price"):
    st.write("Button clicked!")   # Debug line

    input_data = pd.DataFrame([[area, bath, balcony, location]], 
                              columns=['Carpet Area','Bathroom','Balcony','location'])

    st.write("Input Data:", input_data)  # Debug

    prediction = model.predict(input_data)

    st.success(f"Estimated Price: ₹ {prediction[0]*100000:,.2f}")