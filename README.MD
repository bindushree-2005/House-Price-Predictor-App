🏠 House Price Prediction App 

📌 Project Overview

This project is a Machine Learning-based House Price Prediction System that estimates property prices based on key features like area, number of bathrooms, balconies, and location.

The application is built with an interactive web interface using Streamlit, allowing users to input house details and receive real-time price predictions.
__________________________________________________________________________________________________________________________________________________________________________________________________
🤖 Machine Learning Model

This project uses Linear Regression, a supervised learning algorithm used for predicting continuous values.

✔ Why Linear Regression?

Simple and efficient

Suitable for price prediction

Works well with numerical data

Easy to interpret
___________________________________________________________________________________________________________________________________________________________________________________________________________
🌐 Web Application

The app is developed using Streamlit, which provides a fast and user-friendly interface.
_______________________________________________________________________________________________________________________________________________________________________________________________________________
✨ Features

📥 User input for house details

⚡ Real-time price prediction

📊 Clean and interactive UI

💰 Displays estimated house price instantly

📊 Input Features
_____________________________________________________________________________________________________________________________________________________________________________________________________
The model uses the following inputs:

Carpet Area (sqft)

Number of Bathrooms

Number of Balconies

Location (encoded)
________________________________________________________________________________________________________________________________________________________________________________________________________
⚙️ Data Preprocessing

Removed unwanted symbols (₹, sqft, etc.)

Converted text data into numeric format

Handled missing values

Encoded categorical variables (location)
____________________________________________________________________________________________________________________________________________________________________________________________________________
🚀 How to Run the Project Locally

1️⃣ Clone the Repository

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run the Application

streamlit run app.py

4️⃣ Open in Browser

http://localhost:8501
__________________________________________________________________________________________________________________________________________________________________________________________________________
🌍 Live Demo

https://house-price-predictor-app-7bjtkk5yg24zcfvkcfr6db.streamlit.app/
____________________________________________________________________________________________________________________________________________________________________________________________________________
📂 Project Structure

house-price-prediction/

│── app.py

│── house_data.csv

│── requirements.txt

│── README.md
________________________________________________________________________________________________________________________________________________________________________________________________________________
✅ Advantages

Easy to use

Fast predictions

Real-world application

Beginner-friendly ML project
___________________________________________________________________________________________________________________________________________________________________________________________________________________
⚠️ Limitations

Accuracy depends on dataset quality

Limited features used

Basic encoding for location
_________________________________________________________________________________________________________________________________________________________________________________________________________________
🔮 Future Enhancements

Add more features (floor, furnishing, etc.)

Improve model accuracy

Add dropdown for location

Deploy on cloud for public access
___________________________________________________________________________________________________________________________________________________________________________________________________________________
🎓 Conclusion

This project demonstrates how machine learning can be applied to real-world problems like house price prediction and how it can be integrated with a web interface for better usability.
___________________________________________________________________________________________________________________________________________________________________________________________________________________
Author

Bindushree S Y
_________________________________________________________________________________________________________________________________________________________________________________________________________________
⭐ Acknowledgment

Thanks to open-source libraries and tools that made this project possible.
