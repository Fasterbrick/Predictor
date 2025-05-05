import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import pyodbc

# --- Configuration ---
DATABASE_CONNECTION_STRING = r"Driver={ODBC Driver 17 for SQL Server};Server=192.168.0.1,1433;Database=master;UID=user;PWD=password;"
TABLE_NAME = "BTCUSDminutes"
LOOKBACK_PERIOD = 1000
PREDICTION_PERIOD = 30
MIN_TRAINING_DATA_POINTS = LOOKBACK_PERIOD + PREDICTION_PERIOD
EXTRA_MINUTES_FOR_TRAINING = 500
TOTAL_MINUTES_TO_FETCH = LOOKBACK_PERIOD + PREDICTION_PERIOD + EXTRA_MINUTES_FOR_TRAINING

# --- Device Configuration ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Fetching (unchanged) ---
def fetch_historical_data(conn_string, table_name, num_minutes):
    print(f"Attempting to fetch last {num_minutes} minutes from database...")
    conn = None
    cursor = None
    try:
        conn = pyodbc.connect(conn_string)
        cursor = conn.cursor()
        query = f"SELECT TOP {num_minutes} [time], [open], [high], [low], [close], tick_volume, spread, real_volume FROM dbo.[{table_name}] ORDER BY [time] DESC;"
        cursor.execute(query)
        rows = cursor.fetchall()
        if not rows:
            print("No data fetched from database.")
            return None
        column_names = [column[0] for column in cursor.description]
        df = pd.DataFrame.from_records(rows, columns=column_names)
        df = df.sort_values(by='time').reset_index(drop=True)
        print(f"Successfully fetched {len(df)} rows.")
        if len(df) < num_minutes:
            print(f"Warning: Only fetched {len(df)} minutes, requested {num_minutes}.")
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
    if data is None or data.empty:
        print("No data provided for preprocessing.")
        return None, None, None
    dataset = data['close'].values.reshape(-1, 1)
    if np.isnan(dataset).any() or np.isinf(dataset).any():
        print("Error: Data contains NaN or infinite values after selecting 'close' price.")
        return None, None, None
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    X, y = [], []
    for i in range(len(scaled_data) - lookback - prediction_period + 1):
        X.append(scaled_data[i:(i + lookback), 0])
        y.append(scaled_data[(i + lookback):(i + lookback + prediction_period), 0])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # (samples, time_steps, features)
    print(f"Prepared {X.shape[0]} samples for training.")
    return X, y, scaler

# --- Model Definition ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=60, num_layers=1, output_size=PREDICTION_PERIOD):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out[:, -1, :])  # Take last time step
        out = self.linear(out)
        return out

# --- Training Loop ---
def train_model(model, X_train, y_train, epochs=300, batch_size=64):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    best_loss = float('inf')
    patience = 15
    trigger_times = 0

    print("Starting model training...")
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            trigger_times = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        scheduler.step(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

    model.load_state_dict(torch.load('best_model.pth'))
    print("Model training finished.")
    return history

# --- Prediction Function ---
def predict_next_n_minutes(model, last_lookback_data, scaler, prediction_period):
    if model is None or last_lookback_data is None or scaler is None:
        print("Cannot make prediction: Model, data, or scaler is missing.")
        return None
    if len(last_lookback_data) != LOOKBACK_PERIOD:
        print(f"Error: Prediction input data must have exactly {LOOKBACK_PERIOD} minutes, but got {len(last_lookback_data)}.")
        return None
    last_lookback_scaled = scaler.transform(last_lookback_data['close'].values.reshape(-1, 1))
    if np.isnan(last_lookback_scaled).any() or np.isinf(last_lookback_scaled).any():
        print("Error: Prediction input data contains NaN or infinite values after scaling.")
        return None
    X_predict = torch.tensor(last_lookback_scaled, dtype=torch.float32).view(1, -1, 1).to(device)
    model.eval()
    with torch.no_grad():
        predicted_scaled_tensor = model(X_predict)
    predicted_scaled = predicted_scaled_tensor.cpu().numpy()
    predicted_original_scale = scaler.inverse_transform(predicted_scaled)
    print("Prediction complete.")
    return predicted_original_scale[0]

# --- Main Execution ---
if __name__ == "__main__":
    historical_data = fetch_historical_data(DATABASE_CONNECTION_STRING, TABLE_NAME, TOTAL_MINUTES_TO_FETCH)
    if historical_data is not None:
        print("\n--- Fetched Data Summary ---")
        print(f"Number of rows fetched: {len(historical_data)}")
        print("Columns:", historical_data.columns.tolist())
        print("\nMissing values per column:")
        print(historical_data.isnull().sum())
        print("\n'close' column statistics:")
        print(historical_data['close'].describe())
        print("--------------------------")

        if historical_data['close'].isnull().any() or np.isinf(historical_data['close']).any():
            print("Error: Fetched data contains NaN or infinite values in 'close' column.")
        elif len(historical_data) >= MIN_TRAINING_DATA_POINTS:
            training_data = historical_data.iloc[:-PREDICTION_PERIOD].copy()
            prediction_input_data = historical_data.tail(LOOKBACK_PERIOD).copy()
            X_train, y_train, scaler = prepare_data_for_lstm(training_data, LOOKBACK_PERIOD, PREDICTION_PERIOD)
            if X_train is not None and y_train is not None and scaler is not None:
                model = LSTMModel().to(device)
                train_model(model, X_train, y_train, epochs=300, batch_size=64)
                predicted_prices = predict_next_n_minutes(model, prediction_input_data, scaler, PREDICTION_PERIOD)
                if predicted_prices is not None:
                    print(f"\n--- Predicted 'close' prices for the next {PREDICTION_PERIOD} minutes ---")
                    last_known_time = historical_data['time'].iloc[-1]
                    print(f"Prediction based on data up to: {last_known_time}")
                    for i, price in enumerate(predicted_prices):
                        predicted_time = pd.to_datetime(last_known_time) + pd.Timedelta(minutes=i+1)
                        print(f"Minute {i+1} ({predicted_time.strftime('%H:%M')}): {price:.2f}")
                else:
                    print("Prediction failed.")
            else:
                print("Data preparation failed.")
        else:
            print(f"Fetched data is less than the minimum required for training.")
    else:
        print("Failed to fetch historical data.")
