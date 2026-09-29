import pandas as pd
import numpy as np

def main():
    print("=== Bike Dataset Preprocessing ===")
    
    # 1. Load CSV and report initial row count
    df = pd.read_csv('../data/indian_bikes_dataset.csv')
    initial_count = len(df)
    print(f"Original row count: {initial_count}")
    
    # 2. Remove exact duplicates and report
    duplicates = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    final_count = len(df)
    print(f"Duplicate rows removed: {duplicates}")
    print(f"Row count after removing duplicates: {final_count}")
    
    # 3. Clean text columns by trimming unnecessary whitespace
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
    
    # 4. Ensure price, kms_driven, age, and power are numeric
    numeric_cols = ['price', 'kms_driven', 'age', 'power']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # 5. Preserve original categorical values (already preserved)
    
    # 6. Validate invalid or suspicious values
    suspicious = df[
        (df['price'] < 1000) | (df['price'] > 5000000) |
        (df['kms_driven'] < 0) | (df['kms_driven'] > 500000) |
        (df['age'] < 0) | (df['age'] > 50) |
        (df['power'] < 50) | (df['power'] > 2500)
    ]
    print(f"Suspicious rows found: {len(suspicious)}")
    if len(suspicious) > 0:
        print("Sample of suspicious rows:")
        print(suspicious[['bike_name', 'price', 'kms_driven', 'age', 'power']].head())
    
    # 7. Check for remaining missing values and duplicates
    missing_values = df.isnull().sum().sum()
    remaining_duplicates = df.duplicated().sum()
    print(f"Missing values after cleaning: {missing_values}")
    print(f"Remaining duplicates: {remaining_duplicates}")
    print("\nFinal column names and data types:")
    print(df.dtypes)
    print("\nPreview of first 5 cleaned rows:")
    print(df.head())
    
    # Save the cleaned dataset
    df.to_csv('../data/cleaned_bikes_dataset.csv', index=False)
    print("\nSaved cleaned dataset to data/cleaned_bikes_dataset.csv\n")

if __name__ == '__main__':
    main()
