import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load your data
df = pd.read_csv("../data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Data Preprocessing
df = df.dropna()  # Drop missing values, you can customize based on your data

# Define features and target
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use ColumnTransformer to apply OneHotEncoder to categorical columns
categorical_cols = ['JobRole', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'MaritalStatus', 'OverTime']
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Create a preprocessor using OneHotEncoder for categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ])

# Define the RandomForest model and combine it with the preprocessor using a pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Train the model
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Title of the Streamlit app
st.title("Employee Attrition Prediction")

# Display the accuracy
st.write(f"Model Accuracy: {accuracy*100:.2f}%")
st.text(f"Classification Report: \n{report}")

# Show the correlation heatmap
st.subheader('Correlation Heatmap')

# Exclude non-numeric columns (categorical data) for correlation matrix
numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(12, 8))
corr_matrix = numeric_df.corr()  # Only calculate correlation on numeric columns
sns.heatmap(corr_matrix, annot=True, fmt='.1f', linewidths=0.5, cmap='coolwarm')
st.pyplot()

# Show data distribution plots
st.subheader('Attrition Distribution')
sns.countplot(x="Attrition", data=df, palette=["lightblue", "pink"], hue="Attrition", legend=False)
st.pyplot()

# Add interactive widget: Display a sample prediction based on user input
st.subheader('Employee Data Input for Prediction')

# Input fields for the user
age = st.number_input('Age', min_value=18, max_value=60, value=36)
job_role = st.selectbox('Job Role', df['JobRole'].unique())
business_travel = st.selectbox('Business Travel', df['BusinessTravel'].unique())
department = st.selectbox('Department', df['Department'].unique())
education_field = st.selectbox('Education Field', df['EducationField'].unique())
gender = st.selectbox('Gender', df['Gender'].unique())
marital_status = st.selectbox('Marital Status', df['MaritalStatus'].unique())
overtime = st.selectbox('Overtime', df['OverTime'].unique())

# Prepare user input data
user_input = pd.DataFrame([[age, job_role, business_travel, department, education_field, gender, marital_status, overtime]],
                          columns=['Age', 'JobRole', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'MaritalStatus', 'OverTime'])

# Ensure user input is encoded in the same way as the training data (e.g., OneHotEncoding)
user_input_encoded = preprocessor.transform(user_input)

# Predict attrition for user input
user_prediction = model.predict(user_input_encoded)
prediction = 'Left' if user_prediction[0] == 1 else 'Stayed'

st.write(f"Prediction for this employee: {prediction}")