import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Function to generate synthetic data
def generate_data():
    iris = load_iris()
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                      columns=iris['feature_names'] + ['target'])
    return df

# Function to train a machine learning model
def train_model(data):
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy

# Main function
def main():
    st.title("Complex Streamlit App")

    # Sidebar with user authentication
    username = st.sidebar.text_input("Username:")
    password = st.sidebar.text_input("Password:", type="password")
    if st.sidebar.button("Login"):
        if username == "your_username" and password == "your_password":
            st.sidebar.success("Login successful!")
            app_content()
        else:
            st.sidebar.error("Invalid credentials")

# App content for authenticated users
def app_content():
    st.subheader("Data Visualization")

    # Generate synthetic data
    data = generate_data()

    # Display the dataset
    st.write("### Dataset:")
    st.write(data)

    # Train machine learning model
    st.write("### Machine Learning Model")
    trained_model, accuracy = train_model(data)
    st.write(f"Accuracy of the model: {accuracy:.2f}")

    # User input for prediction
    st.write("### Make a Prediction")
    sepal_length = st.slider("Sepal Length", float(data['sepal length (cm)'].min()), float(data['sepal length (cm)'].max()))
    sepal_width = st.slider("Sepal Width", float(data['sepal width (cm)'].min()), float(data['sepal width (cm)'].max()))
    petal_length = st.slider("Petal Length", float(data['petal length (cm)'].min()), float(data['petal length (cm)'].max()))
    petal_width = st.slider("Petal Width", float(data['petal width (cm)'].min()), float(data['petal width (cm)'].max()))

    # Make prediction
    if st.button("Predict"):
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = trained_model.predict(input_data)
        st.write(f"Predicted Iris Species: {prediction[0]}")

# Run the app
if __name__ == "__main__":
    main()
