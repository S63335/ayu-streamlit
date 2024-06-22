import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from pyGRNN import GRNN
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# Title of the application
st.title(" TourVis Pro: Predictive Analytics for Tourism 📊 ")

# Main menu bar on the sidebar
st.sidebar.title("Main Menu")

# Add menu items
menu = st.sidebar.radio(
    "Select an option",
    ("Home", "Data Overview", "Model Training", " SVR Prediction", "GRNN Prediction")
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
        df = pd.read_csv('Malaysia-Tourism1.csv')
        st.dataframe(df)

    with col2:
        st.image("Data Animation.gif", width=400)

elif menu == "Model Training":
    # Add your model training code here
    def main():
        st.title(" 1) Inbound Tourism using SVR ")

        # Reading the csv file
        df = pd.read_csv('Malaysia-Tourism1.csv')
        df.isnull().sum()

        data = df.drop(['Date'], axis=1)
        data.head()

        # Time Series Generator
        # Choose input and output
        n_input = 1
        n_output = 1

        # Creating TimeseriesGenerator
        generator = TimeseriesGenerator(data.values, data.values, length=n_input, batch_size=1)

        # Creating DataFrame to store results
        data_ts = pd.DataFrame(columns=['x', 'y'])

        # Storing results from TimeseriesGenerator into DataFrame
        for i in range(len(generator)):
            x, y = generator[i]
            df_gen = pd.DataFrame({'x': x.flatten(), 'y': y.flatten()})
            data_ts = pd.concat([data_ts, df_gen], ignore_index=True)

        # Displaying DataFrame with results
        st.subheader("Inbound Tourism Actual Data")
        st.write(data_ts)

        X = np.array(data_ts['x'])
        Y = np.array(data_ts['y'])

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Plotting
        import matplotlib.pyplot as plt
        st.subheader('Actual Data Graph')
        fig, ax = plt.subplots()
        plt.plot(Y, label='Prediction Value', marker='X')
    
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')
        plt.title('Actual Data')
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)

        X_train = X_train.reshape(-1,1)
        y_train = y_train.reshape(-1,1)

        X_test = X_test.reshape(-1,1)
        y_test = y_test.reshape(-1,1)

        # Scaling Dataset
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)

        X_test_scaled = scaler_X.fit_transform(X_test)
        y_test_scaled = scaler_y.fit_transform(y_test)

        # Initialize and fit the SVR model
        svr_model = SVR(kernel='rbf')
        svr_model.fit(X_train_scaled, y_train_scaled)

        # Metrics for training set
        st.subheader("Scaled Train Data Value")
        y_pred_train = svr_model.predict(X_train_scaled)
        mse_train = mean_squared_error(y_train_scaled, y_pred_train)
        rmse_train = np.sqrt(mse_train)
        mae_train = mean_absolute_error(y_train_scaled, y_pred_train)
        r2_train = r2_score(y_train_scaled, y_pred_train)

        st.write("Mean Squared Error (Train):", mse_train)
        st.write("Root Mean Squared Error (Train):", rmse_train)
        st.write("Mean Absolute Error (Train):", mae_train)
        st.write("R^2 (Train):", r2_train)

        # Inverse transform for actual values
        st.subheader("Non-scaled Train Data Value")
        y_pred_train_inv = scaler_y.inverse_transform(y_pred_train.reshape(-1,1))
        y_train_inv = scaler_y.inverse_transform(y_train_scaled.reshape(-1,1))

        mse_train_inv = mean_squared_error(y_train_inv, y_pred_train_inv)
        rmse_train_inv = np.sqrt(mse_train_inv)
        mae_train_inv = mean_absolute_error(y_train_inv, y_pred_train_inv)
        r2_train_inv = r2_score(y_train_inv, y_pred_train_inv)

        st.write("Mean Squared Error (Train):", mse_train_inv)
        st.write("Root Mean Squared Error (Train):", rmse_train_inv)
        st.write("Mean Absolute Error (Train):", mae_train_inv)
        st.write("R^2 (Train):", r2_train_inv)

        # Plotting actual vs SVR predictions for training set
        fig, ax = plt.subplots()
        plt.plot(y_pred_train_inv, label='Actual Data', marker='o')
        plt.plot(y_train, label='SVR Prediction', marker='x')
        st.subheader('Actual Data vs SVR Prediction (Train) Graph')
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')
        plt.title('Actual Data vs SVR Prediction (Train)')
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)

        # Metrics for test set
        st.subheader("Scaled Test Data Value")
        y_pred_test = svr_model.predict(X_test_scaled)
        mse_test = mean_squared_error(y_test_scaled, y_pred_test)
        rmse_test = np.sqrt(mse_test)
        mae_test = mean_absolute_error(y_test_scaled, y_pred_test)
        r2_test = r2_score(y_test_scaled, y_pred_test)

        st.write("Mean Squared Error (Train):", mse_test)
        st.write("Root Mean Squared Error (Train):", rmse_test)
        st.write("Mean Absolute Error (Train):", mae_test)
        st.write("R^2 (Train):", r2_test)

        # Inverse transform for actual values
        st.subheader("Non-scaled Test Data Value")
        y_pred_test_inv = scaler_y.inverse_transform(y_pred_test.reshape(-1,1))
        y_test_inv = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1))

        mse_test_inv = mean_squared_error(y_test_inv, y_pred_test_inv)
        rmse_test_inv = np.sqrt(mse_test_inv)
        mae_test_inv = mean_absolute_error(y_test_inv, y_pred_test_inv)
        r2_test_inv = r2_score(y_test_inv, y_pred_test_inv)

        st.write("Mean Squared Error (Train):", mse_test_inv)
        st.write("Root Mean Squared Error (Train):", rmse_test_inv)
        st.write("Mean Absolute Error (Train):", mae_test_inv)
        st.write("R^2 (Train):", r2_test_inv)

        # Plotting actual vs SVR predictions for test set
        fig, ax = plt.subplots()
        plt.plot(y_pred_test_inv, label='Actual Data', marker='o')
        plt.plot(y_test, label='SVR Prediction', marker='x')
        st.subheader('Actual Data vs SVR Prediction (Test) Graph')
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')
        plt.title('Actual vs SVR Prediction (Test)')
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)

        x_scaled = scaler_X.fit_transform(X.reshape(-1,1))
        y_pred = svr_model.predict(x_scaled)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1,1))

        # Membuat plot
        plt.plot(Y, label='Actual Data', marker='o')
        plt.plot(y_pred_inv, label='SVR Prediction', marker='x')

        # Menambahkan label sumbu dan judul
        st.subheader('Actual Data vs SVR Prediction Graph')
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')
        plt.title('ACtual vs SVR Prediction')

        # Menambahkan legenda
        plt.legend()

        # Menampilkan plot
        plt.grid(True)
        st.pyplot(fig)


    if __name__ == "__main__":
        main()

    def main():
        st.title(" 2) Inbound Tourism using GRNN ")

        # Reading the csv file
        df = pd.read_csv('Malaysia-Tourism1.csv')
        df.isnull().sum()

        data = df.drop(['Date'], axis=1)
        data.head()

        # Time Series Generator
        # Choose input and output
        n_input = 1
        n_output = 1

        # Creating TimeseriesGenerator
        generator = TimeseriesGenerator(data.values, data.values, length=n_input, batch_size=1)

        # Creating DataFrame to store results
        data_ts = pd.DataFrame(columns=['x', 'y'])

        # Storing results from TimeseriesGenerator into DataFrame
        for i in range(len(generator)):
            x, y = generator[i]
            df_gen = pd.DataFrame({'x': x.flatten(), 'y': y.flatten()})
            data_ts = pd.concat([data_ts, df_gen], ignore_index=True)

        # Displaying DataFrame with results
        st.subheader("Inbound Tourism Actual Data")
        st.write(data_ts)

        import numpy as np

        X = np.array(data_ts['x'])
        Y = np.array(data_ts['y'])

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Plotting
        import matplotlib.pyplot as plt
        st.subheader('Actual Data Graph')
        fig, ax = plt.subplots()
        plt.plot(Y, label='Prediction Value', marker='X')
    
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')
        plt.title('Actual Data')
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)

        X_train = X_train.reshape(-1,1)
        y_train = y_train.reshape(-1,1)

        X_test = X_test.reshape(-1,1)
        y_test = y_test.reshape(-1,1)

        # Scaling Dataset
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)

        X_test_scaled = scaler_X.fit_transform(X_test)
        y_test_scaled = scaler_y.fit_transform(y_test)

        
        from pyGRNN import GRNN

        # Initialize and fit the GRNN model
        grnn_model = GRNN(calibration="None")
        grnn_model.fit(X_train_scaled, y_train_scaled)

        # Prediksi nilai untuk data latih
        y_pred_train = grnn_model.predict(X_train_scaled)

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np

        # Prediksi nilai untuk data latih
        y_pred_train = grnn_model.predict(X_train_scaled)
        st.subheader("Scaled Train Data Value")
        # Menghitung Mean Squared Error (MSE) untuk data latih
        mse_train = mean_squared_error(y_train_scaled, y_pred_train)
        rmse_train = np.sqrt(mse_train)
        mae_train = mean_absolute_error(y_train_scaled, y_pred_train)
        r2_train = r2_score(y_train_scaled, y_pred_train)

        st.write("Mean Squared Error (Train):", mse_train)
        st.write("Root Mean Squared Error (Train):", rmse_train)
        st.write("Mean Absolute Error (Train):", mae_train)
        st.write("R^2 (Train):", r2_train)

        y_pred_train_inv = scaler_y.inverse_transform(y_pred_train.reshape(-1,1))
        y_train_inv = scaler_y.inverse_transform(y_train_scaled.reshape(-1,1))

        st.subheader("Non-scaled Train Data Value")
        # Menghitung Mean Squared Error (MSE) untuk data latih
        mse_train = mean_squared_error(y_train_inv, y_pred_train_inv)
        rmse_train = np.sqrt(mse_train)
        mae_train = mean_absolute_error(y_train_inv, y_pred_train_inv)
        r2_train = r2_score(y_train_inv, y_pred_train_inv)

        st.write("Mean Squared Error (Train):", mse_train)
        st.write("Root Mean Squared Error (Train):", rmse_train)
        st.write("Mean Absolute Error (Train):", mae_train)
        st.write("R^2 (Train):", r2_train)

        import matplotlib.pyplot as plt
        # Membuat plot
        st.subheader("Actual Data vs GRNN Prediction (Train) Graph")
        fig, ax = plt.subplots()
        plt.plot(y_pred_train_inv, label='Actual Data', marker='o')
        plt.plot(y_train, label='GRNN Prediction', marker='x')

        # Menambahkan label sumbu dan judul
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')
        plt.title('Actual Data vs GRNN Prediction (Train) Graph')

        # Menambahkan legenda
        plt.legend()

        # Menampilkan plot
        plt.grid(True)
        st.pyplot(fig)

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np

        # Prediksi nilai untuk data latih
        y_pred_test = grnn_model.predict(X_test_scaled)

        st.subheader("Scaled Test Data Value")
        # Menghitung Mean Squared Error (MSE) untuk data latih
        mse_test = mean_squared_error(y_test_scaled, y_pred_test)
        st.write("Mean Squared Error (Test):", mse_test)

        # Menghitung Root Mean Squared Error (RMSE) untuk data latih
        rmse_test = np.sqrt(mse_train)
        st.write("Root Mean Squared Error (Test):", rmse_test)

        # Menghitung Mean Absolute Error (MAE) untuk data latih
        mae_test = mean_absolute_error(y_test_scaled, y_pred_test)
        st.write("Mean Absolute Error (Test):", mae_test)

        # Menghitung Koefisien Determinasi (R^2) untuk data latih
        r2_test = r2_score(y_test_scaled, y_pred_test)
        st.write("R^2 (Test):", r2_test)

        y_pred_test_inv = scaler_y.inverse_transform(y_pred_test.reshape(-1,1))
        y_test_inv = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1))

        st.subheader("Non-scaled Test Data Value")
        # Menghitung Mean Squared Error (MSE) untuk data latih
        mse_test = mean_squared_error(y_test_inv, y_pred_test_inv)
        st.write("Mean Squared Error (Test):", mse_test)

        # Menghitung Root Mean Squared Error (RMSE) untuk data latih
        rmse_test = np.sqrt(mse_test)
        st.write("Root Mean Squared Error (Test):", rmse_test)

        # Menghitung Mean Absolute Error (MAE) untuk data latih
        mae_test = mean_absolute_error(y_test_inv, y_pred_test_inv)
        st.write("Mean Absolute Error (Test):", mae_test)

        # Menghitung Koefisien Determinasi (R^2) untuk data latih
        r2_test = r2_score(y_test_inv, y_pred_test_inv)
        st.write("R^2 (Test):", r2_test)

        import matplotlib.pyplot as plt
        st.subheader("Actual Data vs GRNN Prediction (Test) Graph")
        fig, ax = plt.subplots()
        # Membuat plot
        plt.plot(y_pred_test_inv, label='Actual Data', marker='o')
        plt.plot(y_test, label='GRNN Prediction', marker='x')

        # Menambahkan label sumbu dan judul
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')
        plt.title('Actual Data vs GRNN Prediction (Test)')

        # Menambahkan legenda
        plt.legend()

        # Menampilkan plot
        plt.grid(True)
        st.pyplot(fig)


        x_scaled = scaler_X.fit_transform(X.reshape(-1,1))
        y_pred = grnn_model.predict(x_scaled)

        import matplotlib.pyplot as plt
        y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1,1))

        st.subheader("Actual Data vs GRNN Prediction Graph")
        fig, ax = plt.subplots()
        # Membuat plot
        plt.plot(Y, label='Actual Data', marker='o')
        plt.plot(y_pred_inv, label='GRNN Prediction', marker='x')

        # Menambahkan label sumbu dan judul
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')
        plt.title('Actual Data vs GRNN Prediction')

        # Menambahkan legenda
        plt.legend()

        # Menampilkan plot
        plt.grid(True)
        st.pyplot(fig)

    if __name__ == "__main__":
        main()

elif menu == " SVR Prediction":

    def main():
        st.title("SVR Inbound Tourism Forecasting")

        # Read data from CSV file
        df = pd.read_csv('Malaysia-Tourism1.csv')

        # Reading the csv file
        df = pd.read_csv('Malaysia-Tourism1.csv')
        df.isnull().sum()

        data = df.drop(['Date'], axis=1)
        data.head()

        # Time Series Generator
        # Choose input and output
        n_input = 1
        n_output = 1

        # Creating TimeseriesGenerator
        generator = TimeseriesGenerator(data.values, data.values, length=n_input, batch_size=1)

        # Creating DataFrame to store results
        data_ts = pd.DataFrame(columns=['x', 'y'])

        # Storing results from TimeseriesGenerator into DataFrame
        for i in range(len(generator)):
            x, y = generator[i]
            df_gen = pd.DataFrame({'x': x.flatten(), 'y': y.flatten()})
            data_ts = pd.concat([data_ts, df_gen], ignore_index=True)

        # Displaying DataFrame with results
        st.subheader("Inbound Tourism Actual Data")
        st.write(data_ts)

        X = data_ts['x'].values.reshape(-1, 1)
        Y = data_ts['y'].values.reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        X_train = X_train.reshape(-1,1)
        y_train = y_train.reshape(-1,1)

        X_test = X_test.reshape(-1,1)
        y_test = y_test.reshape(-1,1)

        # Scaling Dataset
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)

        X_test_scaled = scaler_X.fit_transform(X_test)
        y_test_scaled = scaler_y.fit_transform(y_test)

        # Initialize and fit the SVR model
        svr_model = SVR(kernel='rbf')
        svr_model.fit(X_train_scaled, y_train_scaled.ravel())

        y_pred_train = svr_model.predict(X_train_scaled)
        y_pred_train_inv = scaler_y.inverse_transform(y_pred_train.reshape(-1,1))
        y_train_inv = scaler_y.inverse_transform(y_train_scaled.reshape(-1,1))
        y_pred_test = svr_model.predict(X_test_scaled)
        y_pred_test_inv = scaler_y.inverse_transform(y_pred_test.reshape(-1,1))
        y_test_inv = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1))

        x_scaled = scaler_X.fit_transform(X.reshape(-1,1))
        y_pred = svr_model.predict(x_scaled)

        y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1,1))

        # User input for the number of predicted values
        num_predictions = st.number_input("Enter the number of predicted values (1-50)", min_value=1, max_value=50, value=15)

        # Initial prediction using the last known data point
        last_data_point = X[-1].reshape(1, -1)
        y_pred_scaled = svr_model.predict(scaler_X.transform(last_data_point)).reshape(-1, 1)
        predictions = []

        # Predict 6 new data points
        for _ in range(num_predictions):
            # Reshape and inverse transform the predicted value
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
            predictions.append(y_pred.flatten()[0])
            y_pred_scaled = svr_model.predict(scaler_X.transform(y_pred)).reshape(-1, 1)
        
        predictions = np.array(predictions)

        # Print out the forecast values
        st.subheader("Forecasted Values")
        for i, pred in enumerate(predictions):
            st.write(f"Prediction {i + 1}: {pred}")

        # Plotting the actual data, SVR prediction, and forecasting
        fig, ax = plt.subplots()
        ax.plot(Y, label='Actual Data', marker='o')
        y_pred_inv = scaler_y.inverse_transform(svr_model.predict(scaler_X.transform(X)).reshape(-1, 1))
        ax.plot(y_pred_inv, label='SVR Prediction', marker='x')

        # Plotting the forecasting values
        forecast_index = range(len(Y), len(Y) + len(predictions))
        ax.plot(forecast_index, predictions, color='red', marker='o', label='Forecasting')

        # Adding labels and title
        ax.set_xlabel('Month')
        ax.set_ylabel('Tourism Data')
        ax.set_title('Actual Data vs SVR Prediction & Forecasting')
        ax.legend()
        ax.grid(True)

        # Displaying the plot
        st.pyplot(fig)


    if __name__ == "__main__":
        main()

elif menu == "GRNN Prediction":
    
    def main():
        st.title("GRNN Inbound Tourism Forecasting")

        # Read data from CSV file
        df = pd.read_csv('Malaysia-Tourism1.csv')

        data = df.drop(['Date'], axis=1)
        data.head()

        # Time Series Generator
        # Choose input and output
        n_input = 1
        n_output = 1

        # Creating TimeseriesGenerator
        generator = TimeseriesGenerator(data.values, data.values, length=n_input, batch_size=1)

        # Creating DataFrame to store results
        data_ts = pd.DataFrame(columns=['x', 'y'])

        # Storing results from TimeseriesGenerator into DataFrame
        for i in range(len(generator)):
            x, y = generator[i]
            df_gen = pd.DataFrame({'x': x.flatten(), 'y': y.flatten()})
            data_ts = pd.concat([data_ts, df_gen], ignore_index=True)

        # Displaying DataFrame with results
        st.subheader("Inbound Tourism Actual Data")
        st.write(data_ts)

        X = data_ts['x'].values.reshape(-1, 1)
        Y = data_ts['y'].values.reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        X_train = X_train.reshape(-1,1)
        y_train = y_train.reshape(-1,1)

        X_test = X_test.reshape(-1,1)
        y_test = y_test.reshape(-1,1)

        # Scaling Dataset
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)

        X_test_scaled = scaler_X.fit_transform(X_test)
        y_test_scaled = scaler_y.fit_transform(y_test)

        from pyGRNN import GRNN

        # Initialize and fit the GRNN model
        grnn_model = GRNN(calibration="None")
        grnn_model.fit(X_train_scaled, y_train_scaled)

        y_pred_train = grnn_model.predict(X_train_scaled)
        y_pred_train_inv = scaler_y.inverse_transform(y_pred_train.reshape(-1,1))
        y_train_inv = scaler_y.inverse_transform(y_train_scaled.reshape(-1,1))
        y_pred_test = grnn_model.predict(X_test_scaled)
        y_pred_test_inv = scaler_y.inverse_transform(y_pred_test.reshape(-1,1))
        y_test_inv = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1))

        x_scaled = scaler_X.fit_transform(X.reshape(-1,1))
        y_pred = grnn_model.predict(x_scaled)

        y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1,1))

        # User input for the number of predicted values
        num_predictions = st.number_input("Enter the number of predicted values (1-50)", min_value=1, max_value=50, value=15)

        # Initial prediction using the last known data point
        last_data_point = X[-1].reshape(1, -1)
        y_pred_scaled = grnn_model.predict(scaler_X.transform(last_data_point)).reshape(-1, 1)
        predictions = []

        # Predict 6 new data points
        for _ in range(num_predictions):
            # Reshape and inverse transform the predicted value
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
            predictions.append(y_pred.flatten()[0])
            y_pred_scaled = grnn_model.predict(scaler_X.transform(y_pred)).reshape(-1, 1)
        
        predictions = np.array(predictions)

        # Print out the forecast values
        st.subheader("Forecasted Values")
        for i, pred in enumerate(predictions):
            st.write(f"Prediction {i + 1}: {pred}")

        # Plotting the actual data, SVR prediction, and forecasting
        fig, ax = plt.subplots()
        ax.plot(Y, label='Actual Data', marker='o')
        y_pred_inv = scaler_y.inverse_transform(grnn_model.predict(scaler_X.transform(X)).reshape(-1, 1))
        ax.plot(y_pred_inv, label='GRNN Prediction', marker='x')

        # Plotting the forecasting values
        forecast_index = range(len(Y), len(Y) + len(predictions))
        ax.plot(forecast_index, predictions, color='red', marker='o', label='Forecasting')

        # Adding labels and title
        ax.set_xlabel('Month')
        ax.set_ylabel('Tourism Data')
        ax.set_title('Actual Data vs GRNN Prediction & Forecasting')
        ax.legend()
        ax.grid(True)

        # Displaying the plot
        st.pyplot(fig)


    if __name__ == "__main__":
        main()
