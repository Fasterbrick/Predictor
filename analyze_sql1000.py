import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pyodbc # Keep pyodbc import for database interaction

# --- Configuration ---
# Use the connection string from your previous code
DATABASE_CONNECTION_STRING = r"Driver={ODBC Driver 17 for SQL Server};Server=192.168.0.1,1433;Database=master;UID=user;PWD=password;"
TABLE_NAME = "BTCUSDminutes" # Your table name
LOOKBACK_PERIOD = 1000 # Use last 1000 minutes as input for prediction
PREDICTION_PERIOD = 30 # Predict the next 30 minutes

# Minimum data required for at least one training sample (LOOKBACK + PREDICTION)
MIN_TRAINING_DATA_POINTS = LOOKBACK_PERIOD + PREDICTION_PERIOD

# Number of extra minutes to fetch for creating more training samples
# A larger number here means more training data, potentially better model
EXTRA_MINUTES_FOR_TRAINING = 500

# Total minutes to fetch from the database
# This includes data for training samples AND the final prediction input
TOTAL_MINUTES_TO_FETCH = LOOKBACK_PERIOD + PREDICTION_PERIOD + EXTRA_MINUTES_FOR_TRAINING

# --- Data Fetching ---
def fetch_historical_data(conn_string, table_name, num_minutes):
    """
    Fetches the last 'num_minutes' of data from the database.

    Args:
        conn_string (str): Database connection string.
        table_name (str): Name of the table.
        num_minutes (int): Number of minutes of historical data to fetch.

    Returns:
        pd.DataFrame: DataFrame containing the historical data,
                      sorted by time ascending, or None if fetching fails or not enough data.
    """
    print(f"Attempting to fetch last {num_minutes} minutes from database...")
    conn = None
    cursor = None
    try:
        conn = pyodbc.connect(conn_string)
        cursor = conn.cursor()

        # Query to fetch the last 'num_minutes' rows ordered by time descending,
        # then order ascending in Python to get oldest first.
        # Adjust column names if 'time' is not the correct ordering column.
        # Ensure you select the columns needed ('time', 'open', 'high', 'low', 'close', etc.)
        query = f"SELECT TOP {num_minutes} [time], [open], [high], [low], [close], tick_volume, spread, real_volume FROM dbo.[{table_name}] ORDER BY [time] DESC;"
        cursor.execute(query)
        rows = cursor.fetchall()

        if not rows:
            print("No data fetched from database.")
            return None

        # Convert rows to DataFrame and sort by time ascending
        # Get column names from cursor description
        column_names = [column[0] for column in cursor.description]
        df = pd.DataFrame.from_records(rows, columns=column_names)
        df = df.sort_values(by='time').reset_index(drop=True)

        print(f"Successfully fetched {len(df)} rows.")
        if len(df) < num_minutes:
             print(f"Warning: Only fetched {len(df)} minutes, requested {num_minutes}.")

        # Check if we have enough data for at least one training sample
        if len(df) < MIN_TRAINING_DATA_POINTS:
             print(f"Error: Not enough data ({len(df)} minutes) to create even one training sample (requires at least {MIN_TRAINING_DATA_POINTS}).")
             return None

        return df

    except pyodbc.Error as e:
        print(f"Database error during data fetching: {e}")
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# --- Data Preprocessing ---
def prepare_data_for_lstm(data, lookback, prediction_period):
    """
    Prepares the data for LSTM training. Creates sequences of lookback_period
    inputs and prediction_period targets.

    Args:
        data (pd.DataFrame): The input time series data (e.g., 'close' prices)
                             to be used for creating training samples.
        lookback (int): The number of past time steps to use as input (X).
        prediction_period (int): The number of future time steps to predict (y).

    Returns:
        tuple: A tuple containing:
               - X (np.ndarray): Input sequences for the LSTM.
               - y (np.ndarray): Output sequences (target values).
               - scaler (MinMaxScaler): The scaler fitted on the data.
    """
    if data is None or data.empty:
        print("No data provided for preprocessing.")
        return None, None, None

    # Use only the 'close' price for this simple example
    # You can extend this to use multiple features ('open', 'high', 'low', etc.)
    dataset = data['close'].values.reshape(-1, 1)

    # --- Check for NaN or infinite values in the dataset ---
    if np.isnan(dataset).any() or np.isinf(dataset).any():
        print("Error: Data contains NaN or infinite values after selecting 'close' price. Cannot proceed with scaling and training.")
        # In a real application, you might implement data cleaning/imputation here.
        return None, None, None
    # --- End Check ---


    # Scale the data - fit the scaler on the training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    X, y = [], []

    # Ensure we have enough data points to create at least one sample
    if len(scaled_data) < lookback + prediction_period:
        print(f"Not enough data points ({len(scaled_data)}) to create samples with lookback {lookback} and prediction period {prediction_period}.")
        return None, None, None

    # Create sequences
    # The loop goes up to the point where the target sequence can be fully formed
    for i in range(len(scaled_data) - lookback - prediction_period + 1):
        # Create input sequence (X): data from index i to i + lookback - 1
        seq_x = scaled_data[i:(i + lookback), 0]
        X.append(seq_x)

        # Create output sequence (y): data from index i + lookback to i + lookback + prediction_period - 1
        seq_y = scaled_data[(i + lookback):(i + lookback + prediction_period), 0]
        y.append(seq_y)

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Reshape X for LSTM [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    print(f"Prepared {X.shape[0]} samples for training.")
    return X, y, scaler

# --- Build LSTM Model ---
def build_lstm_model(lookback, prediction_period):
    """
    Builds a simple LSTM model.

    Args:
        lookback (int): The number of past time steps in the input sequence.
        prediction_period (int): The number of future time steps to predict.

    Returns:
        tensorflow.keras.models.Sequential: The compiled LSTM model.
    """
    model = Sequential()

    # Layer 1: LSTM layer with return_sequences=True for stacking LSTM layers
    model.add(LSTM(units=60, return_sequences=True, input_shape=(lookback, 1))) # Increased units slightly
    model.add(Dropout(0.2)) # Add dropout to prevent overfitting

    # Layer 2: Another LSTM layer
    model.add(LSTM(units=60, return_sequences=False)) # return_sequences=False for the last LSTM layer
    model.add(Dropout(0.2))

    # Output layer: Dense layer to predict 'prediction_period' values
    model.add(Dense(units=prediction_period))

    # Compile the model
    # Added learning_rate and clipvalue for potential stability improvements
    from tensorflow.keras.optimizers import Adam
    model.compile(optimizer=Adam(learning_rate=0.001, clipvalue=1.0), loss='mean_squared_error') # Adam optimizer with gradient clipping

    print("LSTM model built and compiled.")
    model.summary()
    return model

# --- Train Model ---
def train_model(model, X_train, y_train, epochs=200, batch_size=64):
    """
    Trains the LSTM model.

    Args:
        model (tensorflow.keras.models.Sequential): The compiled LSTM model.
        X_train (np.ndarray): Training input data.
        y_train (np.ndarray): Training target data.
        epochs (int): Number of epochs to train.
        batch_size (int): Batch size for training.

    Returns:
        tensorflow.keras.callbacks.History: Training history object.
    """
    if X_train is None or y_train is None:
        print("Cannot train model: Training data is not available.")
        return None

    print("Starting model training...")
    # Add EarlyStopping to stop training when loss stops improving
    early_stopping = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True) # Increased patience

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=1)
    print("Model training finished.")
    return history

# --- Make Prediction ---
def predict_next_n_minutes(model, last_lookback_data, scaler, prediction_period):
    """
    Predicts the next 'prediction_period' values using the trained model.

    Args:
        model (tensorflow.keras.models.Sequential): The trained LSTM model.
        last_lookback_data (pd.DataFrame): The last 'lookback' minutes of data
                                           to be used as the prediction input sequence.
        scaler (MinMaxScaler): The scaler used during preprocessing.
        prediction_period (int): The number of future time steps to predict.

    Returns:
        np.ndarray: The predicted values (in original scale), or None if prediction fails.
    """
    if model is None or last_lookback_data is None or scaler is None:
        print("Cannot make prediction: Model, data, or scaler is missing.")
        return None

    # Ensure we have exactly LOOKBACK_PERIOD data points for prediction input
    if len(last_lookback_data) != LOOKBACK_PERIOD:
        print(f"Error: Prediction input data must have exactly {LOOKBACK_PERIOD} minutes, but got {len(last_lookback_data)}.")
        return None

    # Use the 'close' price and scale it using the *same* scaler fitted on training data
    last_lookback_scaled = scaler.transform(last_lookback_data['close'].values.reshape(-1, 1))

    # --- Check for NaN or infinite values in the scaled prediction input ---
    if np.isnan(last_lookback_scaled).any() or np.isinf(last_lookback_scaled).any():
        print("Error: Prediction input data contains NaN or infinite values after scaling.")
        return None
    # --- End Check ---

    # Reshape for prediction [samples, time steps, features]
    # We have only one sample for prediction
    X_predict = np.array([last_lookback_scaled])
    X_predict = np.reshape(X_predict, (X_predict.shape[0], X_predict.shape[1], 1))

    print(f"Making prediction for the next {prediction_period} minutes...")
    # Make the prediction
    predicted_scaled = model.predict(X_predict)

    # Inverse transform the prediction to the original scale using the *same* scaler
    predicted_original_scale = scaler.inverse_transform(predicted_scaled)

    print("Prediction complete.")
    # The prediction is a 2D array [[price1, price2, ...]], return the inner array
    return predicted_original_scale[0]

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Fetch historical data
    # Fetch enough data for training samples AND the final prediction input
    historical_data = fetch_historical_data(DATABASE_CONNECTION_STRING, TABLE_NAME, TOTAL_MINUTES_TO_FETCH)

    if historical_data is not None:
        # --- Added: Detailed inspection of fetched data ---
        print("\n--- Fetched Data Summary ---")
        print(f"Number of rows fetched: {len(historical_data)}")
        print("Columns:", historical_data.columns.tolist())
        print("\nMissing values per column:")
        print(historical_data.isnull().sum())
        print("\n'close' column statistics:")
        print(historical_data['close'].describe())
        print("--------------------------\n")
        # --- End Added Inspection ---

        # --- Initial check for NaN/Inf in fetched data (kept from previous version) ---
        if historical_data['close'].isnull().any() or np.isinf(historical_data['close']).any():
             print("Error: Fetched historical data contains NaN or infinite values in 'close' column. Cannot proceed.")
        # --- End Initial Check ---
        elif len(historical_data) >= MIN_TRAINING_DATA_POINTS:

            # 2. Split data into training data and prediction input data
            # Training data: all fetched data EXCEPT the last PREDICTION_PERIOD minutes
            # This ensures the target values for training are not part of the final prediction input
            training_data = historical_data.iloc[:-PREDICTION_PERIOD].copy()

            # Prediction input data: the very last LOOKBACK_PERIOD minutes of the fetched data
            # This is the sequence the model will use to predict the next PREDICTION_PERIOD minutes
            prediction_input_data = historical_data.tail(LOOKBACK_PERIOD).copy()

            # 3. Prepare training data for LSTM
            # The scaler will be fitted on this training data
            X_train, y_train, scaler = prepare_data_for_lstm(training_data, LOOKBACK_PERIOD, PREDICTION_PERIOD)

            if X_train is not None and y_train is not None and scaler is not None:
                # 4. Build LSTM Model
                model = build_lstm_model(LOOKBACK_PERIOD, PREDICTION_PERIOD)

                # 5. Train Model
                # Train using the prepared training samples
                train_model(model, X_train, y_train, epochs=300, batch_size=64) # Increased epochs again, adjust as needed

                # 6. Make Prediction
                # Use the trained model, the prediction input data, and the fitted scaler
                predicted_prices = predict_next_n_minutes(model, prediction_input_data, scaler, PREDICTION_PERIOD)

                if predicted_prices is not None:
                    print(f"\n--- Predicted 'close' prices for the next {PREDICTION_PERIOD} minutes ---")
                    # Get the time of the last known data point
                    last_known_time = historical_data['time'].iloc[-1]
                    print(f"Prediction based on data up to: {last_known_time}")

                    # Print the predicted prices with estimated times
                    for i, price in enumerate(predicted_prices):
                        # Assuming each prediction step corresponds to one minute
                        predicted_time = pd.to_datetime(last_known_time) + pd.Timedelta(minutes=i+1)
                        print(f"Minute {i+1} (approx. {predicted_time.strftime('%H:%M')}): {price:.2f}") # Format to 2 decimal places
                else:
                    print("Prediction failed.")

            else:
                print("Data preparation failed. Cannot build or train model.")
        else:
             print(f"Fetched data ({len(historical_data)} minutes) is less than the minimum required for training ({MIN_TRAINING_DATA_POINTS}). Cannot proceed.")
    else:
        print("Failed to fetch historical data to proceed.")
