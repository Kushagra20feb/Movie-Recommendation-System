import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
file_path = '/mnt/data/IMDb Movies India.csv'
df = pd.read_csv(file_path)

# Explore the dataset
print(df.head())
print(df.info())
print(df.describe())

# Handle missing values
# For simplicity, let's drop rows with missing values
df = df.dropna()

# Extract features and target variable
features = ['genre', 'director', 'actors']  # Assuming these columns exist
target = 'rating'  # Assuming this column exists

X = df[features]
y = df[target]

# One-hot encode categorical features
categorical_features = ['genre', 'director', 'actors']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Define the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Optionally, save the model
import joblib
joblib.dump(model, 'movie_rating_predictor.pkl')
