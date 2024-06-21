import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Title of the application
st.title("TourVis Pro: Predictive Analytics for Tourism 📊")

# Main menu bar on the sidebar
st.sidebar.title("Main Menu")

# Add menu items
menu = st.sidebar.radio(
    "Select an option",
    ("Home", "Data Overview", "Model Training", "Predictions", "About")
)

# Display content based on the selected menu item
if menu == "Home":
    st.header("Welcome to SVR & GRNN Web-based App! 👋")

    col1, col2 = st.columns([1, 2])  # Split the page into 1/4 and 3/4

    with col1:
        st.image("Robot Animation.gif", width=200)

    with col2:
        st.sidebar.success("Select a demo above.")

        st.markdown(
            """
        Welcome to our SVR & GRNN Web-Based App! Predict tourist arrivals in Malaysia with precision using Support Vector Regression and Generalized Regression Neural Network technology. Our user-friendly platform empowers businesses and policymakers with accurate forecasting for any selected year. Experience the future of tourism prediction today!
            """
        )

elif menu == "Predictions":

    # Function to load the model
    @st.cache(allow_output_mutation=True)
    def load_trained_model(model_path):
        model = load_model(model_path)
        return model

    def main():
        st.title("Inbound Tourism Forecasting")

        # Read data from CSV file
        df = pd.read_csv('Malaysia-Tourism1.csv')

        # Show data
        st.subheader("Data:")
        st.write(df.head())

        # Check for missing values
        if df.isnull().sum().sum() > 0:
            st.warning("Data contains missing values. Please handle them before proceeding.")
            return

        # Drop the Date column
        data = df.drop(['Date'], axis=1)

        # Time Series Generator
        n_input = 1
        n_output = 1
        generator = TimeseriesGenerator(data.values, data.values, length=n_input, batch_size=1)

        # Create a DataFrame for the time series data
        data_ts = pd.DataFrame(columns=['x', 'y'])
        for i in range(len(generator)):
            x, y = generator[i]
            df_temp = pd.DataFrame({'x': x.flatten(), 'y': y.flatten()})
            data_ts = pd.concat([data_ts, df_temp], ignore_index=True)

        st.header("Train Data")
        st.write(data_ts)

        # Split Data
        X = data_ts['x'].values.reshape(-1, 1)
        y = data_ts['y'].values.reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scaling Dataset
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)

        # User input for the number of predicted values
        num_predictions = st.number_input("Enter the number of predicted values (1-50)", min_value=1, max_value=50, value=15)

        # Create a new figure
        fig, ax = plt.subplots()

        # Plotting the actual data
        ax.plot(y, label='Actual Data', marker='o')

        # Display "Select Model:" as a header
        st.subheader("Select Model:")
        # Model selection using radio button
        model_selection = st.radio("", ("Support Vector Regression (SVR)", "General Regression Neural Network (GRNN)"))

        predictions = []

        if model_selection == "Support Vector Regression (SVR)":
            # Initialize and fit the SVR model
            svr_model = SVR(kernel='rbf')
            svr_model.fit(X_train_scaled, y_train_scaled.ravel())

            # Initial prediction using the last known data point
            last_data_point = X[-1].reshape(1, -1)
            y_pred_scaled = svr_model.predict(scaler_X.transform(last_data_point))

            # Predict new data points based on user input
            for _ in range(num_predictions):
                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
                predictions.append(y_pred.flatten()[0])
                next_data_point = y_pred.reshape(1, -1)
                y_pred_scaled = svr_model.predict(scaler_X.transform(next_data_point))

            predictions = np.array(predictions)

            # Plotting the SVR prediction
            ax.plot(range(len(y), len(y) + num_predictions), predictions, label='SVR Prediction', marker='x')

        elif model_selection == "General Regression Neural Network (GRNN)":
            st.warning("GRNN implementation is not available in this version.")
            return

        # Plotting the forecasting values
        for i in range(num_predictions):
            ax.plot(len(y) + i, predictions[i], color='red', marker='o', label='Forecasting' if i == 0 else None)

        # Adding labels and title
        ax.set_xlabel('Month')
        ax.set_ylabel('Tourism Data')
        ax.set_title('Actual Data vs Model Prediction & Forecasting')

        # Adding legend
        ax.legend()

        # Displaying the plot in Streamlit
        st.pyplot(fig)

    if __name__ == "__main__":
        main()

elif menu == "About":
    st.header("About")
    st.write("This section contains information about the application.")
    st.write("""
    This application was created to forecast tourism numbers using various machine learning models.
    """)
