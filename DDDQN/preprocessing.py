import pandas as pd
import matplotlib.pyplot as plt

def preprocess_data(df):
    """
    Preprocess OHLCV DataFrame by filling missing values and adding technical indicators.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data (Open, High, Low, Close, Volume)
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame with added indicators
    """
    if df.isnull().sum().sum() > 0:
        print(f"Warning: Initial missing values found: {df.isnull().sum().sum()}")
        df = df.ffill() 
        df = df.bfill()
    else:
        print("No missing values found, skipping fill operations.")

    df['SMA7'] = df['Close'].rolling(window=7).mean()  # 7-day SMA
    df['SMA25'] = df['Close'].rolling(window=25).mean() # 25-day SMA
    df['EMA12'] = df['Close'].ewm(span=12).mean() # 12-day EMA
    df['EMA26'] = df['Close'].ewm(span=26).mean() # 26-day EMA
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9).mean()   # 9-day EMA of MACD
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean() # 14-day gain
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean() # 14-day loss
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Daily_Return'] = df['Close'].pct_change()
    df['Norm_Close'] = df['Close'] / df['Close'].iloc[0]
    
    df.dropna(inplace=True)
    
    return df

def visualize_data(df):
    """
    Visualize cryptocurrency data and indicators in one combined figure.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame
    """
    fig, axs = plt.subplots(5, 1, figsize=(15, 40))
    
    # 1. Price vs Time
    axs[0].plot(df.index, df['Close'], label='Close Price', color='blue')
    axs[0].set_title('Price vs Time')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Price')
    axs[0].legend()
    
    # 2. Price and Moving Averages
    axs[1].plot(df.index, df['Close'], label='Close Price', color='blue')
    axs[1].plot(df.index, df['SMA7'], label='SMA 7', color='orange')
    axs[1].plot(df.index, df['SMA25'], label='SMA 25', color='green')
    axs[1].set_title('Price and Moving Averages')
    axs[1].set_ylabel('Price')
    axs[1].legend()
    
    # 3. MACD
    axs[2].plot(df.index, df['MACD'], label='MACD', color='purple')
    axs[2].plot(df.index, df['Signal'], label='Signal', color='red')
    axs[2].bar(df.index, df['MACD'] - df['Signal'], alpha=0.3, label='Histogram', color='gray')
    axs[2].set_title('MACD')
    axs[2].legend()
    
    # 4. RSI
    axs[3].plot(df.index, df['RSI'], label='RSI', color='magenta')
    axs[3].axhline(y=70, color='r', linestyle='--', alpha=0.5)
    axs[3].axhline(y=30, color='g', linestyle='--', alpha=0.5)
    axs[3].set_title('RSI')
    axs[3].set_ylim(0, 100)
    axs[3].legend()
    
    # 5. Daily Returns
    axs[4].plot(df.index, df['Daily_Return'], label='Daily Return', color='cyan')
    axs[4].set_title('Daily Returns')
    axs[4].set_ylabel('Daily Return')
    axs[4].legend()

    plt.tight_layout()
    plt.show()

