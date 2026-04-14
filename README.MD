🏠 House Price Prediction Project

📌 Project Overview

This project is a Machine Learning-based House Price Prediction System that estimates property prices based on key features like area, number of bathrooms, balconies, and location.

The application is built with an interactive web interface using Streamlit, allowing users to input house details and receive real-time price predictions.
___________________________________________________________________________________________________________________________________________________________________________________________
🤖 Machine Learning Model

This project uses Linear Regression, a supervised learning algorithm used for predicting continuous values.

✔ Why Linear Regression?

Simple and efficient

Suitable for price prediction

Works well with numerical data

Easy to interpret
______________________________________________________________________________________________________________________________________________________________________________________________________
🌐 Web Application
The app is developed using Streamlit, which provides a fast and user-friendly interface.

✨ Features

📥 User input for house details

⚡ Real-time price prediction

📊 Clean and interactive UI

💰 Displays estimated house price instantly

📊 Input Features
_________________________________________________________________________________________________________________________________________________________________________________________
The model uses the following inputs:

Carpet Area (sqft)
Number of Bathrooms
Number of Balconies
Location (encoded)📊 Input Features  
| Feature | Description |
|--------|------------|
| 📐 Carpet Area | Size of the house (sqft) |
| 🚿 Bathrooms | Number of bathrooms |
| 🌇 Balconies | Number of balconies |
| 📍 Location | Encoded location value |
________________________________________________________________________________________________________________________________________________________________________________________________

⚙️ Data Preprocessing  

✔ Cleaned unwanted symbols (₹, sqft)  

✔ Converted text to numeric values  

✔ Handled missing data  

✔ Encoded categorical variables  
________________________________________________________________________________________________________________________________________________________________________________________________

 📂 Project Structure

house-price-prediction/
│
├── app.py                # Main Streamlit application (UI + model)
├── small_data.csv        # Cleaned and reduced dataset used for training
├── requirements.txt      # List of required Python libraries
├── README.md             # Project documentation

________________________________________________________________________________________________________________________________________________________________________________________________

🚀 How to Run the Project Locally
1️⃣ Clone the Repository

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Application
streamlit run app.py

4️⃣ Open in Browser
http://localhost:8501
________________________________________________________________________________________________________________________________________________________________________________________________________
🌍 Live Demo:https://house-price-predictor-app-7bjtkk5yg24zcfvkcfr6db.streamlit.app/
__________________________________________________________________________________________________________________________________________________________________________________________________________
📂 Project Structure

house-price-prediction/
│── app.py
│── house_data.csv
│── requirements.txt
│── README.md
_________________________________________________________________________________________________________________________________________________________________________________________________________
✅ Advantages

✔ Beginner-friendly

✔ Fast predictions

✔ Real-world use case

✔ Interactive UI
____________________________________________________________________________________________________________________________________________________________________________________________________________
⚠️ Limitations

❗ Accuracy depends on dataset quality

❗ Limited features

❗ Basic location encoding
___________________________________________________________________________________________________________________________________________________________________________________________________________
🔮 Future Enhancements

🚀 Add more features (floor, furnishing)

🎯 Improve model accuracy

📍 Use dropdown for location

🌍 Enhance UI design
_________________________________________________________________________________________________________________________________________________________________________________________________________
🎓 Conclusion

This project demonstrates how machine learning can be applied to real-world problems like house price prediction and how it can be integrated with a web interface for better usability.
__________________________________________________________________________________________________________________________________________________________________________________________________________
⭐ Acknowledgment

Thanks to open-source libraries and tools that made this project possible.
_________________________________________________________________________________________________________________________________________________________________________________________________________
Author

Bindushree S Y

