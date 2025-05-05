import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split # Keep for splitting indices if needed
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import joblib
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Check for available accelerators
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS (Metal Performance Shaders) on Apple Silicon")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA on {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU for computations")
    return device

# Set the device globally
DEVICE = get_device()

# Import pyodbc with error handling
try:
    import pyodbc
    PYODBC_AVAILABLE = True
except ImportError:
    PYODBC_AVAILABLE = False
    print("pyodbc not available. Will use sample data instead of database connection.")

# Connect to the database using ODBC
def connect_to_database(conn_string=None):
    if not PYODBC_AVAILABLE:
        print("pyodbc module not available. Cannot connect to database.")
        return None

    try:
        if conn_string is None:
            # Replace with your connection details
            conn_string = r"Driver={ODBC Driver 17 for SQL Server};Server=192.168.0.1,1433;Database=master;UID=;PWD="

        connection = pyodbc.connect(conn_string)
        print("Database connection successful")
        return connection
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

# Create sample data (can increase size for testing the memory efficiency)
def create_sample_data(num_samples=500000): # Increased sample size for memory testing
    print("Creating sample Bitcoin price data for testing...")
    start_date = datetime(2025, 4, 1)
    dates = [start_date + timedelta(minutes=i) for i in range(num_samples)]
    base_price = 80000
    prices = []
    current_price = base_price

    trend_factors = [0.0001, -0.00005, 0.00015, -0.0001]
    volatility_factors = [0.0010, 0.0025, 0.0015, 0.0020]

    for i in range(num_samples):
        period = (i // 100000) % 4 # Adjust period length for larger sample
        trend = trend_factors[period]
        volatility = np.random.normal(0, volatility_factors[period])

        if np.random.random() < 0.001: # Reduced shock probability
            volatility *= 5

        current_price = current_price * (1 + volatility + trend)

        daily_range = current_price * np.random.uniform(0.0005, 0.002) # Reduced daily range
        open_price = current_price
        close_price = current_price * (1 + np.random.normal(0, 0.0005)) # Reduced noise
        high_price = max(open_price, close_price) + daily_range * np.random.uniform(0.1, 0.5)
        low_price = min(open_price, close_price) - daily_range * np.random.uniform(0.1, 0.5)

        low_price = min(low_price, open_price, close_price)
        high_price = max(high_price, open_price, close_price)

        volume_factor = 1 + 2 * abs(close_price - open_price) / (open_price + 1e-9)

        prices.append({
            'time': dates[i],
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'tick_volume': int(np.random.exponential(20) * volume_factor),
            'spread': int(np.random.normal(3500, 50)),
            'real_volume': int(np.random.exponential(10) * volume_factor),
            'candle_type': 'bullish' if close_price > open_price else ('bearish' if close_price < open_price else 'neutral'),
            'range': high_price - low_price
        })

    df = pd.DataFrame(prices)
    print(f"Sample data created successfully. Shape: {df.shape}")
    print("Sample data (first 3 rows):")
    print(df.head(3))
    return df

# Fetch data from the database
def fetch_bitcoin_data(connection):
    query = """
    SELECT [time], [open], [high], [low], [close], [tick_volume], [spread], [real_volume],
           [candle_type], [range]
    FROM BTCUSDminutes
    ORDER BY [time]
    """
    try:
        df = pd.read_sql(query, connection)
        print(f"Data fetched successfully. Shape: {df.shape}")
        print("Sample data (first 3 rows):")
        print(df.head(3))
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Feature engineering
def create_features(df):
    """Add technical indicators and other features to the dataframe"""
    df = df.copy() # Operate on a copy

    df['price_diff'] = df['close'] - df['open']
    df['body_ratio'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9) # Avoid division by zero
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']

    df['volume_ratio'] = df['tick_volume'] / (df['tick_volume'].rolling(10).mean().fillna(1) + 1e-9) # Avoid division by zero

    # Use .copy() to avoid SettingWithCopyWarning with rolling operations
    df['ma5'] = df['close'].rolling(5).mean().fillna(method='bfill').copy()
    df['ma10'] = df['close'].rolling(10).mean().fillna(method='bfill').copy()
    df['ma20'] = df['close'].rolling(20).mean().fillna(method='bfill').copy()

    df['momentum'] = df['close'] - df['close'].shift(5).fillna(method='bfill')
    df['volatility'] = df['close'].rolling(10).std().fillna(method='bfill').copy()

    df['close_to_high'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-9)
    df['close_to_low'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)

    return df

# Preprocess the data - Now returns the DataFrame and scalers
def preprocess_data(df):
    """
    Adds features, handles categorical data, creates target, and fits scalers.
    Returns the processed DataFrame and the fitted scalers.
    """
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])

    df = create_features(df)

    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['day_of_week'] = df['time'].dt.dayofweek
    df['day_of_month'] = df['time'].dt.day
    df['month'] = df['time'].dt.month
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    df['candle_type'] = df['candle_type'].replace('neutral', 'bearish')
    label_encoder = LabelEncoder()
    df['candle_type_encoded'] = label_encoder.fit_transform(df['candle_type'])

    df['next_close'] = df['close'].shift(-1)

    # Drop rows that will have NaN due to shifting or rolling operations
    # The number of NaNs depends on the max window size used (e.g., ma20) and sequence_length
    df = df.dropna().reset_index(drop=True) # Reset index after dropping rows

    # Select features
    features = [
        'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume', 'range',
        'price_diff', 'body_ratio', 'upper_shadow', 'lower_shadow', 'volume_ratio',
        'ma5', 'ma10', 'ma20', 'momentum', 'volatility', 'close_to_high', 'close_to_low',
        'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'day_of_week_sin', 'day_of_week_cos',
        'is_weekend', 'candle_type_encoded'
    ]

    # Fit scalers on the *entire* processed data before splitting
    scaler_X = StandardScaler()
    scaler_X.fit(df[features].values) # Fit on feature values

    scaler_y = StandardScaler()
    scaler_y.fit(df['next_close'].values.reshape(-1, 1)) # Fit on target values

    print(f"Preprocessed data shape: {df.shape}")

    # Return the dataframe and scalers
    return df, scaler_X, scaler_y, features

# Create PyTorch dataset for sequence data - Lazy Loading
class BitcoinSequenceDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, features: list, target: str, sequence_length: int, scaler_X: StandardScaler, scaler_y: StandardScaler):
        self.dataframe = dataframe
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        # The effective number of sequences we can create
        self.num_sequences = len(self.dataframe) - self.sequence_length

        # Check if there are enough data points to create at least one sequence
        if self.num_sequences <= 0:
             raise ValueError(f"Not enough data to create sequences. Need at least {sequence_length + 1} data points.")


    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # Get the sequence of features and the target value
        # idx ranges from 0 to self.num_sequences - 1
        start_idx = idx
        end_idx = idx + self.sequence_length

        # Get feature sequence (NumPy array)
        feature_sequence = self.dataframe[self.features].values[start_idx:end_idx]

        # Get target value (NumPy array)
        target_value = self.dataframe[self.target].values[end_idx -1] # Target is the next_close corresponding to the end of the sequence

        # Scale the feature sequence
        # Need to reshape the sequence for scaling
        n_timesteps, n_features = feature_sequence.shape
        feature_sequence_scaled = self.scaler_X.transform(feature_sequence.reshape(-1, n_features))
        feature_sequence_scaled = feature_sequence_scaled.reshape(n_timesteps, n_features) # Reshape back

        # Scale the target value
        target_value_scaled = self.scaler_y.transform(target_value.reshape(-1, 1)).flatten()[0]

        # Convert to PyTorch tensors and move to the device
        X_tensor = torch.tensor(feature_sequence_scaled, dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(target_value_scaled, dtype=torch.float32).to(DEVICE)

        return X_tensor, y_tensor

# Define models (remain the same)
class BitcoinLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(BitcoinLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class BitcoinHybridModel(nn.Module):
    def __init__(self, input_size, sequence_length, hidden_size=128, num_layers=2, dropout=0.2):
        super(BitcoinHybridModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)

        self.cnn_output_size = 128 * (sequence_length // 2)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size + self.cnn_output_size, 128)
        self.bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_features = lstm_out[:, -1, :]

        x_cnn = x.transpose(1, 2)
        cnn_out = self.relu(self.conv1(x_cnn))
        cnn_out = self.pool(cnn_out)
        cnn_out = self.relu(self.conv2(cnn_out))
        cnn_out = cnn_out.view(cnn_out.size(0), -1)

        combined = torch.cat((lstm_features, cnn_out), dim=1)
        combined = self.dropout(combined)

        out = self.relu(self.fc1(combined))
        out = self.bn(out)
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.fc3(out)

        return out

# Train the model (Minor changes for new Dataset handling)
def train_model(train_dataset: Dataset, test_dataset: Dataset, input_size: int, sequence_length: int, batch_size=64, epochs=100, patience=15, model_type="hybrid"):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if model_type == "lstm":
        model = BitcoinLSTMModel(input_size=input_size).to(DEVICE)
        print("Using LSTM model architecture")
    else:
        model = BitcoinHybridModel(input_size=input_size, sequence_length=sequence_length).to(DEVICE)
        print("Using Hybrid CNN-LSTM model architecture")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)

    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    best_model_state = None
    no_improve_count = 0

    print(f"Training for {epochs} epochs with early stopping (patience={patience})...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for inputs, targets in progress_bar:
            # Data is moved to DEVICE within the Dataset's __getitem__ and DataLoader batching
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() # Use .item() to prevent graph accumulation

            progress_bar.set_postfix({"batch_loss": loss.item()})

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0.0
        with torch.no_grad(): # Crucial for memory
            for inputs, targets in test_loader:
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)
                test_loss += loss.item() # Use .item()

        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        scheduler.step(test_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model.state_dict().copy() # Copy state dict
            no_improve_count = 0
            print(f"✅ New best model saved! Loss: {best_test_loss:.6f}")
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

        if DEVICE.type == 'mps':
            torch.mps.empty_cache() # Clear cache after each epoch

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plot_file = 'loss_curve.png'
    plt.savefig(plot_file)
    plt.close() # Close plot
    print(f"Loss curve saved to {plot_file}")

    if DEVICE.type == 'mps':
        torch.mps.empty_cache()

    return model, train_losses, test_losses

# Make predictions (Minor changes for new Dataset handling)
def make_predictions(model: nn.Module, test_dataset: Dataset, scaler_y: StandardScaler):
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    all_preds = []
    all_actuals = [] # Also collect actuals from the dataset for evaluation

    with torch.no_grad(): # Crucial for memory
        for inputs, targets in test_loader:
            # inputs and targets are already on the device
            outputs = model(inputs).squeeze()
            all_preds.append(outputs.cpu().numpy()) # Move to CPU and numpy
            all_actuals.append(targets.cpu().numpy()) # Move to CPU and numpy

    pred_scaled = np.concatenate(all_preds).flatten()
    y_test_scaled = np.concatenate(all_actuals).flatten() # Use actuals collected from dataset

    y_pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    y_actual = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    mse = np.mean((y_pred - y_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_actual))
    mape = np.mean(np.abs((y_actual - y_pred) / (y_actual + 1e-9))) * 100 # Avoid division by zero

    print("\nPrediction Performance Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")

    plt.figure(figsize=(14, 7))
    plot_points = min(500, len(y_actual)) # Increased points for visualization
    plt.plot(y_actual[:plot_points], label='Actual', color='blue', alpha=0.7)
    plt.plot(y_pred[:plot_points], label='Predicted', color='red', alpha=0.7)
    plt.title('Bitcoin Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.fill_between(
        range(plot_points),
        y_pred[:plot_points],
        y_actual[:plot_points],
        alpha=0.2,
        color='gray',
        label='Error'
    )
    pred_plot_file = 'prediction_plot.png'
    plt.savefig(pred_plot_file)
    plt.close() # Close plot
    print(f"Prediction plot saved to {pred_plot_file}")

    plt.figure(figsize=(12, 7))
    errors = y_pred - y_actual
    plt.hist(errors, bins=100, density=True, alpha=0.7, color='skyblue', edgecolor='black') # Increased bins
    mu, std = np.mean(errors), np.std(errors)
    x = np.linspace(mu - 4*std, mu + 4*std, 200) # Increased points for normal curve
    p = 1/(std * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * std**2))
    plt.plot(x, p, 'r-', linewidth=2, label=f'Normal Dist. (μ={mu:.2f}, σ={std:.2f})')
    plt.title('Prediction Error Distribution')
    plt.xlabel('Error Value ($)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Zero Error')
    stats_text = f"Mean Error: ${mu:.2f}\nStd Dev: ${std:.2f}\nRMSE: ${rmse:.2f}"
    plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                 va='top', fontsize=10)
    error_hist_file = 'error_histogram.png'
    plt.savefig(error_hist_file)
    plt.close() # Close plot
    print(f"Error histogram saved to {error_hist_file}")

    if DEVICE.type == 'mps':
        torch.mps.empty_cache()

    return y_pred, y_actual, {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'mean_error': mu,
        'std_error': std
    }

# Calculate trading performance (Remains the same, operates on numpy arrays)
def calculate_trading_performance(y_pred, y_actual, initial_capital=10000, trade_size_percent=0.1):
    actions = []
    capital = initial_capital
    btc_holdings = 0
    stop_loss_percent = 0.02
    take_profit_percent = 0.03
    stop_loss_price = None
    take_profit_price = None
    entry_price = None
    position = None

    for i in range(1, len(y_pred)):
        predicted_direction = y_pred[i] > y_actual[i-1]
        actual_direction = y_actual[i] > y_actual[i-1]
        current_price = y_actual[i]
        previous_price = y_actual[i-1]

        prediction_confidence = 1 - abs((y_pred[i] - current_price) / (current_price + 1e-9))

        action = "HOLD"
        trade_size = capital * trade_size_percent

        if position == 'long' and entry_price is not None:
            if current_price <= stop_loss_price:
                capital += btc_holdings * current_price
                btc_holdings = 0
                action = "SELL (STOP LOSS)"
                position = None
                entry_price = None
                stop_loss_price = None
                take_profit_price = None
            elif current_price >= take_profit_price:
                capital += btc_holdings * current_price
                btc_holdings = 0
                action = "SELL (TAKE PROFIT)"
                position = None
                entry_price = None
                stop_loss_price = None
                take_profit_price = None
        elif position == 'short' and entry_price is not None:
             simulated_short_value_change = (entry_price - current_price) * (trade_size / (entry_price + 1e-9))
             if current_price >= stop_loss_price:
                 capital += trade_size + simulated_short_value_change
                 action = "BUY (STOP LOSS)"
                 position = None
                 entry_price = None
                 stop_loss_price = None
                 take_profit_price = None
             elif current_price <= take_profit_price:
                 capital += trade_size + simulated_short_value_change
                 action = "BUY (TAKE PROFIT)"
                 position = None
                 entry_price = None
                 stop_loss_price = None
                 take_profit_price = None

        if action == "HOLD":
            if predicted_direction and prediction_confidence > 0.6:
               if position is None:
                   if capital >= trade_size:
                       btc_bought = trade_size / (previous_price + 1e-9)
                       btc_holdings += btc_bought
                       capital -= trade_size
                       action = "BUY"
                       position = 'long'
                       entry_price = previous_price
                       stop_loss_price = entry_price * (1 - stop_loss_percent)
                       take_profit_price = entry_price * (1 + take_profit_percent)
            elif not predicted_direction and prediction_confidence > 0.6:
                if position is None:
                     if capital >= trade_size:
                         capital -= trade_size
                         action = "SELL (SHORT)"
                         position = 'short'
                         entry_price = previous_price
                         stop_loss_price = entry_price * (1 + stop_loss_percent)
                         take_profit_price = entry_price * (1 - take_profit_percent)

        portfolio_value = capital
        if btc_holdings > 0:
            portfolio_value += btc_holdings * current_price
        elif position == 'short' and entry_price is not None:
             simulated_short_value_change = (entry_price - current_price) * (trade_size / (entry_price + 1e-9))
             portfolio_value = capital + trade_size + simulated_short_value_change

        correct = predicted_direction == actual_direction

        actions.append({
            'time_step': i,
            'predicted_price': y_pred[i],
            'actual_price': current_price,
            'predicted_direction': "UP" if predicted_direction else "DOWN",
            'actual_direction': "UP" if actual_direction else "DOWN",
            'prediction_confidence': prediction_confidence,
            'correct': correct,
            'action': action,
            'capital': capital,
            'btc_holdings': btc_holdings,
            'portfolio_value': portfolio_value,
            'position': position
        })

    df_actions = pd.DataFrame(actions)

    if len(df_actions) == 0:
         print("No trading actions recorded.")
         return {
             'final_value': initial_capital, 'profit_loss': 0.0, 'profit_loss_percent': 0.0,
             'prediction_accuracy': 0.0, 'win_rate': 0.0, 'sharpe_ratio': np.nan,
             'max_drawdown': 0.0, 'number_of_trades': 0, 'actions_summary': pd.DataFrame()
         }


    final_value = df_actions['portfolio_value'].iloc[-1]
    profit_loss = final_value - initial_capital
    profit_loss_percent = (profit_loss / initial_capital) * 100

    prediction_accuracy = df_actions['correct'].mean() * 100

    if len(df_actions) > 1:
        df_actions['log_returns'] = np.log(df_actions['portfolio_value'] / df_actions['portfolio_value'].shift(1))
        daily_returns = df_actions['log_returns'].dropna()
        if daily_returns.std() > 1e-9: # Check for zero standard deviation
             annualization_factor = 60 * 24 * 252
             sharpe_ratio = np.sqrt(annualization_factor) * (daily_returns.mean() / daily_returns.std())
        else:
            sharpe_ratio = np.nan
    else:
        sharpe_ratio = np.nan

    portfolio_values = df_actions['portfolio_value']
    if len(portfolio_values) > 1:
        cumulative_max = portfolio_values.cummax()
        drawdowns = 1 - portfolio_values / (cumulative_max + 1e-9)
        max_drawdown = drawdowns.max() * 100
    else:
        max_drawdown = 0.0

    trades_df = df_actions[df_actions['action'].str.contains('BUY|SELL')].copy()
    if len(trades_df) > 0:
        closed_trades = trades_df[trades_df['action'].str.contains('STOP LOSS|TAKE PROFIT')]
        if len(closed_trades) > 0:
             profitable_closes = closed_trades[closed_trades['action'].str.contains('TAKE PROFIT')]
             win_rate = len(profitable_closes) / len(closed_trades) * 100
        else:
             win_rate = 0.0
    else:
        win_rate = 0.0

    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.plot(df_actions['portfolio_value'], label='Portfolio Value', color='blue')
    plt.axhline(y=initial_capital, color='r', linestyle='--', label='Initial Capital')
    plt.title('Trading Simulation: Portfolio Value Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    for i, row in df_actions.iterrows():
        if 'BUY' in row['action']:
            plt.scatter(i, row['portfolio_value'], color='green', marker='^', s=50, alpha=0.7, label='_nolegend_')
        elif 'SELL' in row['action']:
            plt.scatter(i, row['portfolio_value'], color='red', marker='v', s=50, alpha=0.7, label='_nolegend_')

    plt.subplot(2, 1, 2)
    plt.plot(drawdowns * 100, color='red', label='Drawdown %')
    plt.axhline(y=max_drawdown, color='k', linestyle='--',
                label=f'Max Drawdown: {max_drawdown:.2f}%')
    plt.title('Portfolio Drawdown')
    plt.xlabel('Time Steps')
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    trading_plot_file = 'trading_performance.png'
    plt.savefig(trading_plot_file)
    plt.close()
    print(f"Trading performance plot saved to {trading_plot_file}")

    if DEVICE.type == 'mps':
        torch.mps.empty_cache()

    performance_summary = {
        'final_value': final_value,
        'profit_loss': profit_loss,
        'profit_loss_percent': profit_loss_percent,
        'prediction_accuracy': prediction_accuracy,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'number_of_trades': len(trades_df),
        'actions_summary': df_actions[['time_step', 'action', 'correct', 'portfolio_value']].head(10)
    }
    return performance_summary

# Predict the next price movement (Minor changes for using scaler)
def predict_price_movement(model: nn.Module, scaler_X: StandardScaler, scaler_y: StandardScaler, latest_sequence_raw: np.ndarray, features: list, sequence_length: int):
    model.eval()

    # Ensure the input sequence has the correct shape (sequence_length, num_features)
    if latest_sequence_raw.shape != (sequence_length, len(features)):
        raise ValueError(f"latest_sequence_raw must have shape ({sequence_length}, {len(features)})")

    # Scale the latest sequence
    latest_sequence_scaled = scaler_X.transform(latest_sequence_raw)

    # Convert to tensor and move to device, add batch dimension
    input_tensor = torch.tensor(latest_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_scaled = model(input_tensor).cpu().numpy()

    prediction = scaler_y.inverse_transform(pred_scaled)[0][0]

    # Get the current price (the close price of the last timestep in the raw input sequence)
    current_price_raw = latest_sequence_raw[-1, features.index('close')]
    current_price = current_price_raw # No need to inverse scale as it's from raw data

    if prediction > current_price:
        movement = "UP"
        percentage_change = ((prediction - current_price) / (current_price + 1e-9)) * 100
    else:
        movement = "DOWN"
        percentage_change = ((current_price - prediction) / (current_price + 1e-9)) * 100

    # Simplified confidence based on the magnitude of the predicted change
    # This is a heuristic; a real confidence metric would be more involved.
    confidence = max(0, 100 - abs(percentage_change) * 5) # Heuristic: higher change -> lower confidence

    return prediction, movement, percentage_change, confidence

# Real-time trading strategy (Remains the same)
def trading_strategy(predicted_price, current_price, movement, confidence, threshold=0.6): # Increased threshold
    if current_price == 0:
        price_diff_percent = 0.0
    else:
        price_diff_percent = abs((predicted_price - current_price) / current_price) * 100

    if movement == "UP" and confidence > threshold * 100:
        if price_diff_percent > 0.5:
            return "STRONG BUY", "High confidence in significant upward movement"
        elif price_diff_percent > 0.1:
            return "BUY", "Moderate upward movement expected"
        else:
            return "HOLD", "Predicted movement is small or confidence is moderate"
    elif movement == "DOWN" and confidence > threshold * 100:
        if price_diff_percent > 0.5:
            return "STRONG SELL", "High confidence in significant downward movement"
        elif price_diff_percent > 0.1:
            return "SELL", "Moderate downward movement expected"
        else:
            return "HOLD", "Predicted movement is small or confidence is moderate"
    else:
        return "HOLD", "No clear strong signal or low confidence"


# Save and load model (Remains the same)
def save_model(model, scaler_X, scaler_y, features, sequence_length, filename='bitcoin_prediction_model'):
    os.makedirs('models', exist_ok=True)
    model.cpu()
    torch.save(model.state_dict(), f'models/{filename}.pth')
    model.to(DEVICE)

    joblib.dump({
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'features': features,
        'sequence_length': sequence_length,
        'model_class': model.__class__.__name__
    }, f'models/{filename}_metadata.joblib')
    print(f"Model and metadata saved to models/{filename}")

def load_model(filename='bitcoin_prediction_model'):
    try:
        metadata = joblib.load(f'models/{filename}_metadata.joblib')
        if metadata['model_class'] == 'BitcoinLSTMModel':
            model = BitcoinLSTMModel(input_size=len(metadata['features']))
        else:
            model = BitcoinHybridModel(
                input_size=len(metadata['features']),
                sequence_length=metadata['sequence_length']
            )
        model.load_state_dict(torch.load(f'models/{filename}.pth', map_location='cpu'))
        model.to(DEVICE)
        model.eval()
        if DEVICE.type == 'mps':
            torch.mps.empty_cache()
        return model, metadata['scaler_X'], metadata['scaler_y'], metadata['features'], metadata['sequence_length']
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None, None

# Main function with refactored data handling
def main():
    os.makedirs('results', exist_ok=True)
    if DEVICE.type == 'mps':
        torch.mps.empty_cache()

    try:
        print("=" * 80)
        print(f"Bitcoin Price Prediction Model using {DEVICE}")
        print("=" * 80)

        connection = connect_to_database()
        if connection is None:
            print("Failed to establish database connection. Using sample data instead.")
            df = create_sample_data(num_samples=500000) # Use larger sample
        else:
            df = fetch_bitcoin_data(connection)
            connection.close()
            del connection

        if df is None or df.empty:
            print("No data available. Exiting.")
            return

        # Preprocess the data - now returns DataFrame and scalers
        processed_df, scaler_X, scaler_y, features = preprocess_data(df)

        # Delete raw dataframe after preprocessing
        del df
        # Optional: Clear MPS cache after preprocessing
        if DEVICE.type == 'mps':
            torch.mps.empty_cache()

        sequence_length = 60 # Define sequence length

        # Split the processed DataFrame into train and test sets based on indices
        # This avoids creating large copies of the dataframes/arrays
        test_size = 0.2 # 20% for testing
        # Ensure enough data for train and test after accounting for sequence length
        if len(processed_df) < sequence_length + 2:
             print(f"Not enough data for training and testing with sequence length {sequence_length}.")
             return

        # Determine the split point for the dataframe based on the number of available sequences
        num_sequences = len(processed_df) - sequence_length
        train_sequences_end = int(num_sequences * (1 - test_size))

        # Indices in the *original* processed_df corresponding to the end of sequences
        train_indices_end_in_df = train_sequences_end + sequence_length
        # Test data starts right after the last training data point needed for a sequence
        test_indices_start_in_df = train_indices_end_in_df - sequence_length


        # Create Dataset instances using slices of the *same* processed_df
        # The Dataset will handle getting the correct sequences based on indices
        train_dataset = BitcoinSequenceDataset(
            dataframe=processed_df.iloc[:train_indices_end_in_df].copy(), # Use a copy for the train split
            features=features,
            target='next_close',
            sequence_length=sequence_length,
            scaler_X=scaler_X,
            scaler_y=scaler_y
        )

        test_dataset = BitcoinSequenceDataset(
            dataframe=processed_df.iloc[test_indices_start_in_df:].copy(), # Use a copy for the test split
            features=features,
            target='next_close',
            sequence_length=sequence_length,
            scaler_X=scaler_X,
            scaler_y=scaler_y
        )

        print(f"Number of training sequences: {len(train_dataset)}")
        print(f"Number of testing sequences: {len(test_dataset)}")

        # Delete the large processed_df after creating dataset views (optional, but good practice)
        del processed_df
        if DEVICE.type == 'mps':
            torch.mps.empty_cache()


        model_choice = 'train' # Or 'load'

        if model_choice == 'train':
            input_size = len(features)
            model, train_losses, test_losses = train_model(
                train_dataset, test_dataset,
                input_size=input_size,
                sequence_length=sequence_length,
                batch_size=128,
                epochs=100,
                patience=20,
                model_type="hybrid"
            )
            save_model(model, scaler_X, scaler_y, features, sequence_length)

            # Delete training-specific data structures after training
            del train_dataset, train_losses, test_losses

        else: # model_choice == 'load'
            model, scaler_X, scaler_y, features, sequence_length = load_model()
            if model is None:
                print("Failed to load model. Exiting.")
                return
            # If loading, we still need the test_dataset, scaler_y, features, sequence_length

        # Make predictions using the test_dataset
        # The function now takes the dataset directly
        y_pred, y_actual, metrics = make_predictions(model, test_dataset, scaler_y)

        # Delete test_dataset, y_pred, y_actual after predictions and metrics are done
        del test_dataset, y_pred, y_actual
        if DEVICE.type == 'mps':
            torch.mps.empty_cache()


        # Evaluate trading performance
        # This will require y_pred and y_actual. We should pass the raw actual prices used for prediction.
        # Let's regenerate y_actual from the test_dataset for the trading simulation.
        # A more memory-efficient way would be to save y_actual during make_predictions.
        # For now, let's use the y_actual returned by make_predictions (which we deleted, oops).
        # Let's modify make_predictions to return y_pred and y_actual (numpy arrays).
        # ... (Already done in the refactored make_predictions) ...
        # We need to ensure y_pred and y_actual are available here. Let's bring back the `del` after trading_performance calculation.

        # Re-calling make_predictions to get y_pred and y_actual for trading simulation
        # This is slightly inefficient but keeps the flow clear. In a real system, pass these results.
        # Or, modify make_predictions to return y_pred and y_actual and keep them until after trading_performance.

        # Let's return y_pred and y_actual from make_predictions and pass them to calculate_trading_performance
        # (This is already handled in the make_predictions refactoring).

        # Need to ensure the test_dataset is available to make_predictions
        # If loading, we need to re-create the test_dataset slice.
        # This highlights the benefit of keeping the processed_df or saving the test data separately.

        # --- Let's adjust the flow to keep processed_df or test data until needed ---
        # If training, processed_df is deleted. If loading, it's never created in main.
        # We need the raw data for the *last* sequence prediction as well.

        # --- Revised Flow ---
        # 1. Load raw data (df)
        # 2. Preprocess data (processed_df, scalers, features)
        # 3. Split processed_df indices or keep slices for train/test datasets
        # 4. Create train/test BitcoinSequenceDataset instances referencing the slices/df
        # 5. Train or Load model
        # 6. Make predictions using test_dataset -> returns y_pred, y_actual (numpy)
        # 7. Calculate trading performance using y_pred, y_actual (numpy)
        # 8. Predict next price movement using the *last raw sequence* from processed_df and scalers.

        # Need processed_df or at least the test slice of it available for make_predictions
        # And need the last part of processed_df for the final prediction.

        # Let's keep processed_df until after the trading performance calculation.

        connection = connect_to_database()
        if connection is None:
            print("Failed to establish database connection. Using sample data instead.")
            # Increased sample size again for more robust memory test
            df = create_sample_data(num_samples=1000000)
        else:
            df = fetch_bitcoin_data(connection)
            connection.close()
            del connection

        if df is None or df.empty:
            print("No data available. Exiting.")
            return

        processed_df, scaler_X, scaler_y, features = preprocess_data(df)
        del df # Delete raw df early
        if DEVICE.type == 'mps':
            torch.mps.empty_cache()

        sequence_length = 60

        if len(processed_df) < sequence_length + 2:
             print(f"Not enough data for sequences with length {sequence_length}.")
             return

        num_sequences = len(processed_df) - sequence_length
        test_size = 0.2
        train_sequences_end = int(num_sequences * (1 - test_size))

        # Indices in the *original* processed_df
        # Train data is from index 0 up to (but not including) the start of the test sequences
        # Test sequences start after the last training sequence ends.
        # The last timestep of the last training sequence is at index:
        train_last_timestep_idx = train_sequences_end + sequence_length -1
        # The first timestep of the first test sequence is at index:
        test_first_timestep_idx = train_last_timestep_idx - sequence_length + 1


        # Create dataset instances referencing slices
        train_dataset = BitcoinSequenceDataset(
            dataframe=processed_df.iloc[:train_last_timestep_idx + 1].copy(), # Include the last timestep needed for the last train sequence
            features=features, target='next_close', sequence_length=sequence_length,
            scaler_X=scaler_X, scaler_y=scaler_y
        )

        test_dataset = BitcoinSequenceDataset(
            dataframe=processed_df.iloc[test_first_timestep_idx:].copy(), # Start from the first timestep of the first test sequence
            features=features, target='next_close', sequence_length=sequence_length,
            scaler_X=scaler_X, scaler_y=scaler_y
        )

        print(f"Number of training sequences: {len(train_dataset)}")
        print(f"Number of testing sequences: {len(test_dataset)}")


        model_choice = 'train'

        if model_choice == 'train':
            input_size = len(features)
            model, train_losses, test_losses = train_model(
                train_dataset, test_dataset, input_size=input_size, sequence_length=sequence_length,
                batch_size=128, epochs=100, patience=20, model_type="hybrid"
            )
            save_model(model, scaler_X, scaler_y, features, sequence_length)
            del train_dataset, train_losses, test_losses


        else: # model_choice == 'load'
            model, scaler_X, scaler_y, features, sequence_length = load_model()
            if model is None:
                print("Failed to load model. Exiting.")
                # Need to delete the datasets if we loaded and can't proceed
                del train_dataset, test_dataset
                return

        # Make predictions using the test_dataset
        y_pred, y_actual, metrics = make_predictions(model, test_dataset, scaler_y)

        # Calculate trading performance using y_pred and y_actual (numpy arrays)
        trading_performance = calculate_trading_performance(y_pred, y_actual)

        # Delete prediction results after trading calculation
        del y_pred, y_actual, metrics
        if DEVICE.type == 'mps':
            torch.mps.empty_cache()


        # Print detailed trading performance
        print("\n===== Trading Performance Summary =====")
        print(f"Initial Capital: $10,000.00")
        print(f"Final Portfolio Value: ${trading_performance['final_value']:.2f}")
        print(f"Profit/Loss: ${trading_performance['profit_loss']:.2f} ({trading_performance['profit_loss_percent']:.2f}%)")
        print(f"Prediction Accuracy: {trading_performance['prediction_accuracy']:.2f}%")
        if not np.isnan(trading_performance['sharpe_ratio']):
             print(f"Sharpe Ratio: {trading_performance['sharpe_ratio']:.4f}")
        else:
             print("Sharpe Ratio: N/A (Insufficient data)")
        print(f"Maximum Drawdown: {trading_performance['max_drawdown']:.2f}%")
        print(f"Number of Trades: {trading_performance['number_of_trades']}")
        print("\nSample of Trading Actions:")
        print(trading_performance['actions_summary'])

        # Predict the next price movement using the *last raw sequence* from the processed data
        print("\n===== Next Price Movement Prediction =====")

        # Get the last sequence_length rows from the *processed_df* for prediction
        # Ensure processed_df is still available or reload/get the tail if deleted
        # Since we kept processed_df until here, we can use it.
        if len(processed_df) < sequence_length:
             print("Not enough data for the next price prediction.")
             # Delete remaining large objects before exiting
             del processed_df, model, scaler_X, scaler_y, features, sequence_length
             if DEVICE.type == 'mps':
                 torch.mps.empty_cache()
             return


        latest_sequence_raw = processed_df[features].values[-sequence_length:]

        predicted_price, movement, percentage_change, confidence = predict_price_movement(
            model, scaler_X, scaler_y, latest_sequence_raw, features, sequence_length
        )

        # Get current price from the last row of the raw latest sequence
        current_price = latest_sequence_raw[-1, features.index('close')]


        print(f"Current close price: ${current_price:.2f}")
        print(f"Predicted next close price: ${predicted_price:.2f}")
        print(f"Predicted movement: {movement} ({percentage_change:.2f}%)")
        print(f"Prediction confidence: {confidence:.2f}%")

        strategy, reason = trading_strategy(predicted_price, current_price, movement, confidence)
        print(f"\nTrading Recommendation: {strategy}")
        print(f"Reason: {reason}")

        # Create strategy visualization
        plt.figure(figsize=(10, 6))
        plt.bar(['Current Price', 'Predicted Price'],
                [current_price, predicted_price],
                color=['blue', 'green' if movement == "UP" else 'red'])
        for i, v in enumerate([current_price, predicted_price]):
            plt.text(i, v + 1, f'${v:.2f}', ha='center')
        arrow_color = 'green' if movement == "UP" else 'red'
        plt.annotate('', xy=(1, predicted_price), xytext=(0, current_price),
                    arrowprops=dict(arrowstyle='->', lw=2, color=arrow_color))
        plt.figtext(0.5, 0.01, f"RECOMMENDATION: {strategy}\n{reason}",
                   ha='center', fontsize=12,
                   bbox=dict(boxstyle='round,pad=1', facecolor='yellow', alpha=0.5))
        plt.title('Bitcoin Price Prediction')
        plt.ylabel('Price ($)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        recommendation_file = 'trading_recommendation.png'
        plt.savefig(recommendation_file)
        plt.close()
        print(f"Trading recommendation chart saved to {recommendation_file}")

        # Delete remaining large objects
        del processed_df, model, scaler_X, scaler_y, features, sequence_length

        if DEVICE.type == 'mps':
            torch.mps.empty_cache()

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()
        if DEVICE.type == 'mps':
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
