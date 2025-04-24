from datetime import datetime, timedelta
import MetaTrader5 as mt5
import pandas as pd
import time
import pyodbc

# Set up pandas display options
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1500)

# --- Configuration ---
# SQL Server connection string
DATABASE_CONNECTION_STRING = r"Driver={ODBC Driver 17 for SQL Server};Server=localhost\NameofDB;Database=master;Trusted_Connection=yes;"
TABLE_NAME = "BTCUSDminutes"
INITIAL_CANDLES = 50000  # Number of candles to fetch at startup

def initialize_mt5():
    """Initialize connection to MetaTrader 5 without GUI"""
    if not mt5.initialize(portable=True):  # Run without GUI
        print("Initialize() failed, error code =", mt5.last_error())
        return False
    print("MetaTrader 5 connected successfully in headless mode")
    return True

def create_database_connection():
    """Create a connection to SQL Server database"""
    try:
        conn = pyodbc.connect(DATABASE_CONNECTION_STRING)
        cursor = conn.cursor()
        return conn, cursor
    except pyodbc.Error as e:
        print(f"Database connection error: {e}")
        return None, None

def create_table(conn, cursor, recreate=True):
    """Create table for BTC USD minute data with proper error handling and escaped column names"""
    try:
        # Drop table if it exists and recreate is True
        if recreate:
            cursor.execute(f"IF OBJECT_ID('dbo.{TABLE_NAME}', 'U') IS NOT NULL DROP TABLE dbo.{TABLE_NAME}")
            conn.commit()
            print(f"Table {TABLE_NAME} dropped")
        
        # Create table
        cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'BTCUSDminutes')
        BEGIN
            CREATE TABLE dbo.BTCUSDminutes (
                [time] DATETIME PRIMARY KEY,
                [open] FLOAT,
                [high] FLOAT,
                [low] FLOAT,
                [close] FLOAT,
                tick_volume INT,
                spread INT,
                real_volume INT
            )
        END
        """)
        conn.commit()
        print(f"Table {TABLE_NAME} created")
        return True
    except pyodbc.Error as e:
        print(f"Error creating table: {e}")
        return False

def format_data(rates_frame):
    """Process the raw MT5 data frame"""
    if rates_frame is None or len(rates_frame) == 0:
        return pd.DataFrame()
        
    # Convert time in seconds to datetime format
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
    
    # Remove 2 hours (timezone adjustment)
    rates_frame['time'] = rates_frame['time'] - pd.Timedelta(hours=2)
    
    return rates_frame

def insert_data(conn, cursor, data_frame):
    """Insert data into the database with escaped column names"""
    if data_frame is None or len(data_frame) == 0:
        print("No data to insert")
        return 0
        
    rows_inserted = 0
    
    for _, row in data_frame.iterrows():
        try:
            # Check if record already exists
            cursor.execute(f"SELECT COUNT(*) FROM dbo.{TABLE_NAME} WHERE [time] = ?", 
                           (row['time'],))
            if cursor.fetchone()[0] == 0:
                cursor.execute(f'''
                INSERT INTO dbo.{TABLE_NAME} 
                ([time], [open], [high], [low], [close], tick_volume, spread, real_volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['time'],
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    int(row['tick_volume']),
                    int(row['spread']),
                    int(row['real_volume'])
                ))
                rows_inserted += 1
                
                # Commit every 100 rows to avoid large transactions
                if rows_inserted % 100 == 0:
                    conn.commit()
                    print(f"Committed {rows_inserted} rows so far...")
        except pyodbc.Error as e:
            print(f"Error inserting data: {e}")
            print(f"Problem row: {row}")
    
    try:
        # Final commit for remaining rows
        conn.commit()
        print(f"Total rows inserted: {rows_inserted}")
        return rows_inserted
    except pyodbc.Error as e:
        print(f"Error committing data: {e}")
        return 0

def get_newest_timestamp(cursor):
    """Get the newest timestamp in database"""
    try:
        cursor.execute(f"SELECT MAX([time]) FROM dbo.{TABLE_NAME}")
        result = cursor.fetchone()[0]
        print(f"Most recent record in database: {result}")
        return result
    except pyodbc.Error as e:
        print(f"Error getting newest timestamp: {e}")
        return None

def fetch_initial_historical_data():
    """Fetch large amount of historical data"""
    print(f"Fetching initial historical data ({INITIAL_CANDLES} candles)...")
    rates = mt5.copy_rates_from_pos("BTCUSD", mt5.TIMEFRAME_M1, 0, INITIAL_CANDLES)
    
    if rates is not None and len(rates) > 0:
        rates_frame = pd.DataFrame(rates)
        rates_frame = format_data(rates_frame)
        
        # Exclude the last candle (potentially unfinished)
        historical_data = rates_frame.iloc[:-1]
        print(f"Processed {len(historical_data)} historical candles (excluded last unfinished candle)")
        
        return historical_data
    else:
        print("Error: No historical data returned from MT5")
        return pd.DataFrame()

def fetch_latest_data():
    """Fetch 2 latest candles, return only the completed one"""
    rates = mt5.copy_rates_from_pos("BTCUSD", mt5.TIMEFRAME_M1, 0, 2)
    
    if rates is not None and len(rates) > 0:
        rates_frame = pd.DataFrame(rates)
        rates_frame = format_data(rates_frame)
        
        # Only return the first candle (completed minute)
        if len(rates_frame) >= 1:
            latest_data = rates_frame.iloc[0:1]
            return latest_data, rates_frame
    
    print("Warning: No latest data returned from MT5")
    return pd.DataFrame(), pd.DataFrame()

def main():
    print("Starting MetaTrader5 BTC data collection with simplified approach...")
    # Initialize MT5
    if not initialize_mt5():
        return
    
    # Create database connection
    conn, cursor = create_database_connection()
    if not conn or not cursor:
        print("Failed to connect to database, exiting.")
        mt5.shutdown()
        return
    
    try:
        # Create table, dropping if it exists
        if not create_table(conn, cursor, recreate=True):
            print("Failed to create or access table, exiting.")
            return
        
        # Fetch and store initial historical data
        print("Fetching initial historical data...")
        historical_data = fetch_initial_historical_data()
        if not historical_data.empty:
            insert_data(conn, cursor, historical_data)
            print(f"Added {len(historical_data)} initial candles to database")
        else:
            print("Failed to fetch initial historical data, exiting.")
            return
        
        # Main loop for continuous updates
        print("Starting continuous data collection...")
        while True:
            # Get current time
            current_time = datetime.now()
            
            # Fetch latest data (2 candles, store only the completed one)
            latest_data, full_data = fetch_latest_data()
            
            if not latest_data.empty:
                # Store the completed candle to database
                insert_data(conn, cursor, latest_data)
                
                # Display information for monitoring
                print("\n--- Data updated at:", current_time, "---")
                print("Latest stored candle (completed minute):")
                print(latest_data)
            else:
                print("\n--- No new data available at:", current_time, "---")
            
            # Calculate seconds until the next whole minute
            seconds_to_next_minute = 60 - current_time.second
            
            # Sleep until the next whole minute
            print(f"Waiting {seconds_to_next_minute} seconds until next update...")
            time.sleep(seconds_to_next_minute)
        
    except KeyboardInterrupt:
        print("\nScript terminated by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close database connection and MT5
        if conn:
            conn.close()
        mt5.shutdown()
        print("MT5 connection closed and database connection closed")

if __name__ == "__main__":
    main()