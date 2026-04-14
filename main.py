import pandas as pd

# Load your original file (inside zip or extracted)
data = pd.read_csv("house_prices.csv")   # or .xlsx if Excel

# Save as CSVouse_data.csv", index=False)

print("CSV created successfully!")

import pandas as pd   # Import pandas library to work with datasets

# Read the CSV file and store it in a variable called 'data'
data = pd.read_csv("house_data.csv")

# Display the first 5 rows of the dataset to check if it loaded correctly
print(data.head())

data = pd.read_csv("house_data.csv")

# Remove extra spaces in column names
data.columns = data.columns.str.strip()

print("Before cleaning:", data.shape)
# Display information about the dataset (columns, data types, non-null values)
print(data.info())

# Check and display the number of missing (null) values in each column
print(data.isnull().sum())
print("After cleaning:", data.shape)

# Convert numeric columns
# Convert to numeric
# Clean numeric columns (remove text like sqft, ₹, etc.)
for col in ['Carpet Area', 'Bathroom', 'Balcony', 'Price (in rupees)', 'Super Area']:
    data[col] = data[col].astype(str).str.replace('[^0-9.]', '', regex=True)
    data[col] = pd.to_numeric(data[col], errors='coerce')
# Drop only required missing values
data = data.dropna(subset=['Carpet Area', 'Bathroom', 'Balcony', 'Price (in rupees)'])

print("After cleaning:", data.shape)
print("Cleaned data shape:", data.shape)



# Keep only top 10 most frequent locations
top_locations = data['location'].value_counts().nlargest(10).index

# Replace other locations with 'Other'
data['location'] = data['location'].apply(lambda x: x if x in top_locations else 'Other')

# Now apply encoding
data = pd.get_dummies(data, columns=['location'], drop_first=True)

data = data.drop([
    'Index', 'Title', 'Description', 'Amount(in rupees)',
    'Status', 'Floor', 'Transaction', 'Furnishing',
    'facing', 'overlooking', 'Society', 'Car Parking',
    'Ownership', 'Dimensions', 'Plot Area'
], axis=1)
# Select input features (independent variables) from the dataset
# These columns will be used to predict the house price
X = data[['Carpet Area', 'Bathroom', 'Balcony']]


# Select target variable (dependent variable)
# This is the value we want to predict
y = data['Price (in rupees)']
print(data.columns)
# Remove extra spaces from column names
data.columns = data.columns.str.strip()

print("Columns:", data.columns)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
) 



 # Import function to split data
from sklearn.linear_model import LinearRegression     # Import Linear Regression model

# Split the dataset into training and testing sets
# test_size=0.2 means:
# 80% data → training, 20% data → testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a Linear Regression model object
model = LinearRegression()

# Train the model using training data
# The model learns the relationship between input features (X) and output (y)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(predictions)
print("X_test shape:", X_test.shape)


print("Accuracy:", model.score(X_test, y_test))

import matplotlib.pyplot as plt

plt.scatter(y_test, predictions)
plt.plot(y_test, y_test)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("House Price Prediction")

plt.show()


import pandas as pd

# training part
imputer.fit(X_train)

# prediction part (AFTER training)
new_house = pd.DataFrame([[1500, 3, 2]],
                         columns=['Carpet Area','Bathroom','Balcony'])

new_house = imputer.transform(new_house)
prediction = model.predict(new_house)


print(prediction)
print("Predictions:", predictions)
print("Accuracy:", model.score(X_test, y_test))

import joblib

model = joblib.load(r"C:\Users\Yadun\OneDrive\Documents\Desktop\internship\house_model.pkl")
joblib.dump(imputer, "imputer.pkl")