import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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
        df = pd.read_csv('Malaysia-Tourism1.csv')
        st.dataframe(df)
        
    with col2:
        st.image("Data Animation.gif", width=400)

elif menu == "Model Training":
    # Add your model training code here
    def main():
        st.title(" 1) Inbound Tourism using SVR ")
        
        # Load and display your dataset here
        st.header("- Actual data")
        df = pd.read_csv('Malaysia-Tourism1.csv')
        st.dataframe(df)
        
        df.isnull().sum()

        data = df.drop(['Date'], axis=1)
        data.head()

        #Time Series Generator
        #Choose input and output
        n_input = 1
        n_output = 1

        # Membuat TimeseriesGenerator
        generator = TimeseriesGenerator(data.values, data.values, length=n_input, batch_size=1)

        # Membuat DataFrame untuk menyimpan hasil
        data_ts = pd.DataFrame(columns=['x', 'y'])

        # Menyimpan hasil dari TimeseriesGenerator ke dalam DataFrame
        for i in range(len(generator)):
            x, y = generator[i]
            df = pd.DataFrame({'x': x.flatten(), 'y': y.flatten()})
            data_ts = pd.concat([data_ts, df], ignore_index=True)
            
        st.header("- Train Data")
        st.write(data_ts)

        #Split Data

        import numpy as np
        
        data_ts[['x', 'y']] = data_ts[['x', 'y']].astype(int)

        X = np.array(data_ts['x'])
        Y = np.array(data_ts['y'])

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_train = X_train.reshape(-1,1)
        y_train = y_train.reshape(-1,1)

        X_test = X_test.reshape(-1,1)
        y_test = y_test.reshape(-1,1)

        #Scaling Dataset
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)

        X_test_scaled = scaler_X.fit_transform(X_test)
        y_test_scaled = scaler_y.fit_transform(y_test)

        # Initialize and fit the SVR model
        svr_model = SVR(kernel='rbf')
        svr_model.fit(X_train_scaled, y_train_scaled)

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np

        st.header("Value of Scale Train Data")
        
        # Prediksi nilai untuk data latih
        y_pred_train = svr_model.predict(X_train_scaled)

        # Menghitung Mean Squared Error (MSE) untuk data latih
        mse_train = mean_squared_error(y_train_scaled, y_pred_train)
        st.write("Mean Squared Error (Train):", mse_train)

        # Menghitung Root Mean Squared Error (RMSE) untuk data latih
        rmse_train = np.sqrt(mse_train)
        st.write("Root Mean Squared Error (Train):", rmse_train)

        # Menghitung Mean Absolute Error (MAE) untuk data latih
        mae_train = mean_absolute_error(y_train_scaled, y_pred_train)
        st.write("Mean Absolute Error (Train):", mae_train)

        # Menghitung Koefisien Determinasi (R^2) untuk data latih
        r2_train = r2_score(y_train_scaled, y_pred_train)
        st.write("R^2 (Train):", r2_train)

        st.header("Value of non Scale Train Data")
        
        y_pred_train_inv = scaler_y.inverse_transform(y_pred_train.reshape(-1,1))
        y_train_inv = scaler_y.inverse_transform(y_train_scaled.reshape(-1,1))

        # Menghitung Mean Squared Error (MSE) untuk data latih
        mse_train = mean_squared_error(y_train_inv, y_pred_train_inv)
        st.write("Mean Squared Error (Train):", mse_train)

        # Menghitung Root Mean Squared Error (RMSE) untuk data latih
        rmse_train = np.sqrt(mse_train)
        st.write("Root Mean Squared Error (Train):", rmse_train)

        # Menghitung Mean Absolute Error (MAE) untuk data latih
        mae_train = mean_absolute_error(y_train_inv, y_pred_train_inv)
        st.write("Mean Absolute Error (Train):", mae_train)

        # Menghitung Koefisien Determinasi (R^2) untuk data latih
        r2_train = r2_score(y_train_inv, y_pred_train_inv)
        st.write("R^2 (Train):", r2_train)
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        # Membuat plot
        plt.plot(y_pred_train_inv, label='Actual Data', marker='o')
        plt.plot(y_train, label='SVR Prediction', marker='x')

        # Menambahkan label sumbu dan judul
        st.header("Actual Data vs SVR Prediction (Train) Graph")
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')

        # Menambahkan legenda
        plt.legend()

        # Menampilkan plot
        plt.grid(True)
        st.pyplot(fig)

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np

        st.header("Value of Scale Test Data")
        
        # Prediksi nilai untuk data latih
        y_pred_test = svr_model.predict(X_test_scaled)

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

        st.header("Value of non Scale Test Data")
        
        y_pred_test_inv = scaler_y.inverse_transform(y_pred_test.reshape(-1,1))
        y_test_inv = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1))

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

        fig, ax = plt.subplots()
        # Membuat plot
        plt.plot(y_pred_test_inv, label='Actual Data', marker='o')
        plt.plot(y_test, label='SVR Prediction', marker='x')

        # Menambahkan label sumbu dan judul
        st.header("Actual Data vs SVR Prediction (Test) Graph")
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')
        
        # Menambahkan legenda
        plt.legend()

        # Menampilkan plot
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
        st.header("Actual vs SVR Prediction Graph")
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')
    
        # Menambahkan legenda
        plt.legend()

        # Menampilkan plot
        plt.grid(True)
        st.pyplot(fig)

    if __name__ == "__main__":
        main()

    def main():
        st.title(" 2) Inbound Tourism using GRNN ")

        #Reading the csv file
        st.header("- Actual Data")
        df = pd.read_csv('Malaysia-Tourism1.csv')
        st.dataframe(df)

        df.isnull().sum()

        data = df.drop(['Date'], axis=1)
        data.head()

        #Time Series Generator
        #Choose input and output
        n_input = 1
        n_output = 1

        # Membuat TimeseriesGenerator
        generator = TimeseriesGenerator(data.values, data.values, length=n_input, batch_size=1)

        # Membuat DataFrame untuk menyimpan hasil
        data_ts = pd.DataFrame(columns=['x', 'y'])

        # Menyimpan hasil dari TimeseriesGenerator ke dalam DataFrame
        for i in range(len(generator)):
            x, y = generator[i]
            df = pd.DataFrame({'x': x.flatten(), 'y': y.flatten()})
            data_ts = pd.concat([data_ts, df], ignore_index=True)

        # Menampilkan DataFrame hasil
        st.header("- Train Data")
        st.write(data_ts)

        #Split Data
        import numpy as np
        data_ts[['x', 'y']] = data_ts[['x', 'y']].astype(int)

        X = np.array(data_ts['x'])
        Y = np.array(data_ts['y'])

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        X_train = X_train.reshape(-1,1)
        y_train = y_train.reshape(-1,1)

        X_test = X_test.reshape(-1,1)
        y_test = y_test.reshape(-1,1)

        #Scaling Dataset
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

        st.header("Value of Scale Train Data")
        
        # Prediksi nilai untuk data latih
        y_pred_train = grnn_model.predict(X_train_scaled)

        # Menghitung Mean Squared Error (MSE) untuk data latih
        mse_train = mean_squared_error(y_train_scaled, y_pred_train)
        st.write("Mean Squared Error (Train):", mse_train)

        # Menghitung Root Mean Squared Error (RMSE) untuk data latih
        rmse_train = np.sqrt(mse_train)
        st.write("Root Mean Squared Error (Train):", rmse_train)

        # Menghitung Mean Absolute Error (MAE) untuk data latih
        mae_train = mean_absolute_error(y_train_scaled, y_pred_train)
        st.write("Mean Absolute Error (Train):", mae_train)

        # Menghitung Koefisien Determinasi (R^2) untuk data latih
        r2_train = r2_score(y_train_scaled, y_pred_train)
        st.write("R^2 (Train):", r2_train)

        st.header("Value of non Scale Train Data")
        
        y_pred_train_inv = scaler_y.inverse_transform(y_pred_train.reshape(-1,1))
        y_train_inv = scaler_y.inverse_transform(y_train_scaled.reshape(-1,1))

        # Menghitung Mean Squared Error (MSE) untuk data latih
        mse_train = mean_squared_error(y_train_inv, y_pred_train_inv)
        st.write("Mean Squared Error (Train):", mse_train)

        # Menghitung Root Mean Squared Error (RMSE) untuk data latih
        rmse_train = np.sqrt(mse_train)
        st.write("Root Mean Squared Error (Train):", rmse_train)

        # Menghitung Mean Absolute Error (MAE) untuk data latih
        mae_train = mean_absolute_error(y_train_inv, y_pred_train_inv)
        st.write("Mean Absolute Error (Train):", mae_train)

        # Menghitung Koefisien Determinasi (R^2) untuk data latih
        r2_train = r2_score(y_train_inv, y_pred_train_inv)
        st.write("R^2 (Train):", r2_train)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        # Membuat plot
        plt.plot(y_pred_train_inv, label='Actual Data', marker='o')
        plt.plot(y_train, label='GRNN Prediction', marker='x')

        # Menambahkan label sumbu dan judul
        st.header("Actual Data vs GRNN Prediction (Train) Graph")
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')

        # Menambahkan legenda
        plt.legend()

        # Menampilkan plot
        plt.grid(True)
        st.pyplot(fig)

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np

        st.header("Value of Scale Test Data")
        
        # Prediksi nilai untuk data latih
        y_pred_test = grnn_model.predict(X_test_scaled)

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

        st.header("Value of non Scale Test Data")
        y_pred_test_inv = scaler_y.inverse_transform(y_pred_test.reshape(-1,1))
        y_test_inv = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1))

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
        fig, ax = plt.subplots()
        # Membuat plot
        plt.plot(y_pred_test_inv, label='Actual Data', marker='o')
        plt.plot(y_test, label='GRNN Prediction', marker='x')

        # Menambahkan label sumbu dan judul
        st.header("Actual Data vs GRNN Prediction (Test) Graph")
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')

        # Menambahkan legenda
        plt.legend()

        # Menampilkan plot
        plt.grid(True)
        st.pyplot(fig)

        x_scaled = scaler_X.fit_transform(X.reshape(-1,1))
        y_pred = grnn_model.predict(x_scaled)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1,1))

        # Membuat plot
        plt.plot(Y, label='Actual Data', marker='o')
        plt.plot(y_pred_inv, label='GRNN Prediction', marker='x')

        # Menambahkan label sumbu dan judul
        st.header("Actual Data vs GRNN Prediction Graph")
        plt.xlabel('Month')
        plt.ylabel('Tourism Data')

        # Menambahkan legenda
        plt.legend()

        # Menampilkan plot
        plt.grid(True)
        st.pyplot(fig)


    if __name__ == "__main__":
        main()

elif menu == "Predictions":

   def main():
    st.title("Inbound Tourism Forecasting")

    # Read data from CSV file
    df = pd.read_csv('Malaysia-Tourism1.csv')

    # Show data
    st.subheader("Data:")
    st.write(df.head())

    df.isnull().sum()

    data = df.drop(['Date'], axis=1)
    data.head()

    # Time Series Generator
    # Choose input and output
    n_input = 1
    n_output = 1

    # Membuat TimeseriesGenerator
    generator = TimeseriesGenerator(data.values, data.values, length=n_input, batch_size=1)

    # Membuat DataFrame untuk menyimpan hasil
    data_ts = pd.DataFrame(columns=['x', 'y'])

    # Menyimpan hasil dari TimeseriesGenerator ke dalam DataFrame
    for i in range(len(generator)):
        x, y = generator[i]
        df = pd.DataFrame({'x': x.flatten(), 'y': y.flatten()})
        data_ts = pd.concat([data_ts, df], ignore_index=True)

    st.header("- Train Data")
    st.write(data_ts)

    # Split Data
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.svm import SVR

    data_ts[['x', 'y']] = data_ts[['x', 'y']].astype(int)

    X = np.array(data_ts['x'])
    Y = np.array(data_ts['y'])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train = X_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)

    X_test = X_test.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Scaling Dataset
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    X_test_scaled = scaler_X.fit_transform(X_test)
    y_test_scaled = scaler_y.fit_transform(y_test)

    # Display "Select Model:" as a header
    st.subheader("Select Model:")
    # Model selection using radio button
    model_selection = st.radio("", ("Support Vector Regression (SVR)", "General Regression Neural Network (GRNN)"))

    if model_selection == "Support Vector Regression (SVR)":
        # Initialize and fit the SVR model
        svr_model = SVR(kernel='rbf')
        svr_model.fit(X_train_scaled, y_train_scaled)

        # User input for prediction
        user_input = st.number_input("Enter the value for prediction:", min_value=1)

        # Perform prediction
        prediction_scaled = svr_model.predict(scaler_X.transform([[user_input]]))
        prediction = scaler_y.inverse_transform(prediction_scaled)

    elif model_selection == "General Regression Neural Network (GRNN)":
        from pyGRNN import GRNN

        # Initialize and fit the GRNN model
        grnn_model = GRNN(calibration="None")
        grnn_model.fit(X_train_scaled, y_train_scaled)

        # User input for prediction
        user_input = st.number_input("Enter the value for prediction:", min_value=1)

        # Perform prediction
        prediction_scaled = grnn_model.predict(scaler_X.transform([[user_input]]))
        prediction = scaler_y.inverse_transform(prediction_scaled)

        # Display the prediction
        st.subheader("Prediction:")
        st.write(prediction)

        # Plotting forecasted and actual values
        plot_predictions(data_ts, prediction)

def plot_predictions(data_ts, prediction):
    plt.figure(figsize=(12, 6))
    plt.plot(data_ts['x'], data_ts['y'], label='Actual')
    plt.scatter(data_ts['x'].iloc[-1], prediction, color='red', label='Forecast')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Forecasting')
    plt.legend()
    st.pyplot()
        
if __name__ == "__main__":
    main()

elif menu == "About":
    st.header("About")
    st.write("This section contains information about the application.")
    st.write("""
    This application was created to forecast tourism numbers using various machine learning models.
    """)

# You can add more functionalities and widgets here based on your needs
