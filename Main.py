import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# Title of the application
st.title(" TourVis Pro: Predictive Analytics for Tourism 📊 ")

# Main menu bar on the sidebar
st.sidebar.title("Main Menu")

# Add menu items
menu = st.sidebar.radio(
    "Select an option",
    ("Home", "Data Overview", "Model Training", "Predictions", "About")
)

# Display content based on the selected menu item
if menu == "Home":
    st.header(" Welcome to SVR & GRNN Web-based App! 👋 ")

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

elif menu == "Data Overview":
    st.header(" Data Overview 📈 ")
    st.write("Here you can see the overview of the data used for forecasting from 2011 to 2023.")

    col1, col2 = st.columns([1, 2])  # Split the page into 1/4 and 3/4

    with col1:
        # Load and display your dataset here
        df = pd.read_csv('Malaysia-Tourism.csv')
        st.dataframe(df)

    with col2:
        st.image("Data Animation.gif", width=400)

elif menu == "Model Training":
    # Add your model training code here
    def main():
        st.title(" 1) Inbound Tourism using SVR ")

        #Reading the csv file
        df = pd.read_csv('Malaysia-Tourism.csv')
    
        st.subheader("Data:")
        st.write(df.head())

        # Konversi kolom 'Date' ke format datetime dengan dayfirst=True
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

        # Set 'Date' sebagai indeks
        df.set_index('Date', inplace=True)

        # Siapkan data untuk SVR
        # Mengubah Date menjadi nilai numerik karena SVR tidak bisa bekerja dengan tipe datetime secara langsung
        df['NumericDate'] = df.index.map(pd.Timestamp.toordinal)

        # Fitur dan target
        X = df['NumericDate'].values.reshape(-1, 1)
        y = df['Actual'].values

        # Normalisasi fitur
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # Inisialisasi model SVR
        svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

        # Fit model SVR
        svr_model.fit(X_scaled, y_scaled)

        # Prediksi pada data aktual
        y_pred_scaled = svr_model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        # Plot hasil prakiraan dan nilai aktual
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Actual'], label='Actual')
        ax.plot(df.index, y_pred, label='SVR Predictions', linestyle='--', color='blue')
        ax.set_xlabel('Date')
        ax.set_ylabel('Tourism Numbers')
        ax.set_title('Inbound Tourism using SVR')
        ax.legend()
        st.pyplot(fig)

        # Hitung MSE, RMSE, dan MAE
        st.subheader(f"Value of MSE, RMSE & MAE:")
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        st.write(f'Mean Squared Error (MSE): {mse}')
        st.write(f'Root Mean Squared Error (RMSE): {rmse}')
        st.write(f'Mean Absolute Error (MAE): {mae}')
        st.write(f'R-squared (R^2): {r2}')

    if __name__ == "__main__":
        main()

    def grnn_predict(X_train, y_train, X_test, sigma=0.1):
        diff = X_train - X_test[:, np.newaxis]
        distance = np.exp(-np.sum(diff ** 2, axis=2) / (2 * sigma ** 2))
        output = np.sum(distance * y_train, axis=1, keepdims=True) / np.sum(distance, axis=1, keepdims=True)
        return output

    def main():
        st.title(" 2) Inbound Tourism using GRNN ")

        #Reading the csv file
        df = pd.read_csv('Malaysia-Tourism.csv')
    

        st.subheader("Data:")
        st.write(df.head())

        # Convert 'Date' column to datetime format with dayfirst=True
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

        # Set 'Date' as the index
        df.set_index('Date', inplace=True)

        # Prepare data for GRNN
        df['NumericDate'] = df.index.map(pd.Timestamp.toordinal)

        # Features and target
        X = df['NumericDate'].values.reshape(-1, 1)
        y = df['Actual'].values

        # Normalize features
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # Predict on actual data
        y_pred_scaled = grnn_predict(X_scaled, y_scaled, X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        # Plot actual and predicted values
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Actual'], label='Actual')
        ax.plot(df.index, y_pred, label='GRNN Predictions', linestyle='--', color='blue')
        ax.set_xlabel('Date')
        ax.set_ylabel('Tourism Numbers')
        ax.set_title('Inbound Tourism using GRNN')
        ax.legend()
        st.pyplot(fig)

        # Calculate and display MSE, RMSE, and MAE
        st.subheader(f"Value of MSE, RMSE & MAE:")
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        st.write(f'Mean Squared Error (MSE): {mse}')
        st.write(f'Root Mean Squared Error (RMSE): {rmse}')
        st.write(f'Mean Absolute Error (MAE): {mae}')
        st.write(f'R-squared (R^2): {r2}')

    if __name__ == "__main__":
        main()

elif menu == "Predictions":

    def grnn_predict(X_train, y_train, X_test, sigma=0.1):
        # Calculate the Gaussian kernel
        diff = X_train - X_test[:, np.newaxis]
        distance = np.exp(-np.sum(diff ** 2, axis=2) / (2 * sigma ** 2))
    
        # Calculate the output using the kernel
        output = np.sum(distance * y_train, axis=1, keepdims=True) / np.sum(distance, axis=1, keepdims=True)
    
        return output

    def main():
        st.title("Inbound Tourism Forecasting")

        # Read data from CSV file
        df = pd.read_csv('Malaysia-Tourism.csv')

        # Show data
        st.subheader("Data:")
        st.write(df.head())

        # Convert 'Date' column to datetime format with dayfirst=True
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

        # Set 'Date' as index
        df.set_index('Date', inplace=True)

        # Prepare data for model
        # Convert Date to numerical value because the model cannot work with datetime type directly
        df['NumericDate'] = df.index.map(pd.Timestamp.toordinal)

        # Features and target
        X = df['NumericDate'].values.reshape(-1, 1)
        y = df['Actual'].values

        # Feature scaling
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # Display "Select Model:" as a header
        st.subheader("Select Model:")
        # Model selection using radio button
        model_selection = st.radio("", ("Support Vector Regression (SVR)", "General Regression Neural Network (GRNN)"))

        if model_selection == "Support Vector Regression (SVR)":
            # Initialize SVR model
            model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

            # Fit the SVR model
            model.fit(X_scaled, y_scaled)

        elif model_selection == "General Regression Neural Network (GRNN)":
            # Predictions on actual data using GRNN
            y_pred_scaled = grnn_predict(X_scaled, y_scaled, X_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

            # Input for the number of months to forecast
            num_months = st.number_input("Enter the number of months to forecast:", min_value=1, max_value=50)

            # Make predictions for the next 'num_months' months
            last_date = df.index[-1]
            next_dates = [last_date + pd.DateOffset(months=i) for i in range(1, num_months + 1)]
            next_numeric_dates = np.array([date.toordinal() for date in next_dates]).reshape(-1, 1)

            # Normalizing the prediction dates
            next_numeric_dates_scaled = scaler_X.transform(next_numeric_dates)

            # Predicting values for the next 'num_months' months using GRNN
            next_predictions_scaled = grnn_predict(X_scaled, y_scaled, next_numeric_dates_scaled)
            next_predictions = scaler_y.inverse_transform(next_predictions_scaled.reshape(-1, 1)).ravel()

            # Plotting forecasted and actual values
            st.subheader("Predictions:")
            plot_predictions(df, y_pred, next_dates, next_predictions, num_months)

            # Print predicted values for the next 'num_months' months
            st.subheader(f"Predictions for the next {num_months} months:")
            for date, pred in zip(next_dates, next_predictions):
                st.write(f"Date: {date.strftime('%Y-%m')}, Predicted: {pred}")

            return

        # Predictions on actual data
        y_pred_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        # Input for the number of months to forecast
        num_months = st.number_input("Enter the number of months to forecast:", min_value=1, max_value=50)

        # Make predictions for the next 'num_months' months
        last_date = df.index[-1]
        next_dates = [last_date + pd.DateOffset(months=i) for i in range(1, num_months + 1)]
        next_numeric_dates = np.array([date.toordinal() for date in next_dates]).reshape(-1, 1)

        # Normalizing the prediction dates
        next_numeric_dates_scaled = scaler_X.transform(next_numeric_dates)

        # Predicting values for the next 'num_months' months
        next_predictions_scaled = model.predict(next_numeric_dates_scaled)
        next_predictions = scaler_y.inverse_transform(next_predictions_scaled.reshape(-1, 1)).ravel()

        # Plotting forecasted and actual values
        st.subheader("Predictions Graph:")
        plot_predictions(df, y_pred, next_dates, next_predictions, num_months)

        # Print predicted values for the next 'num_months' months
        st.subheader(f"Predictions for the next {num_months} months:")
        for date, pred in zip(next_dates, next_predictions):
            st.write(f"Date: {date.strftime('%Y-%m')}, Predicted: {pred}")

    def plot_predictions(df, y_pred, next_dates, next_predictions, num_months):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Actual'], label='Actual')
        ax.plot(df.index, y_pred, label='Predictions', linestyle='--', color='blue')
        ax.plot(next_dates, next_predictions, label='Future Predictions', linestyle='--', color='red')
        ax.set_xlabel('Date')
        ax.set_ylabel('Tourism Numbers')
        ax.set_title('Tourism Forecast')
        ax.legend()
        st.pyplot(fig)
    if __name__ == "__main__":
        main()

elif menu == "About":
    st.header("About")
    st.write("This section contains information about the application.")
    st.write("""
    This application was created to forecast tourism numbers using various machine learning models.
    """)

# You can add more functionalities and widgets here based on your needs