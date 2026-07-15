import pandas as pd

def load_data(path):
    """Load dataset from CSV file"""
    return pd.read_csv(path)

def clean_data(df):
    """Basic data cleaning"""
    df = df.dropna()
    return df

def feature_engineering(df):
    """Create or transform features"""
    # Example placeholder
    return df
