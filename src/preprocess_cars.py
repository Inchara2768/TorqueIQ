import pandas as pd
import numpy as np

def extract_brand(model_name):
    # List of known multi-word car brands in the Indian market
    multi_word_brands = [
        "Land Rover", "Aston Martin", "Rolls Royce", "Rolls-Royce",
        "Mercedes-Benz", "Mercedes Benz", "Hindustan Motors", 
        "Maruti Suzuki", "Force Motors", "Alfa Romeo"
    ]
    
    for brand in multi_word_brands:
        if model_name.upper().startswith(brand.upper()):
            return brand
    
    # Default to first word
    return model_name.split(" ")[0]

def main():
    print("=== Car Dataset Preprocessing ===")
    
    # 1. Load CSV and report initial row count
    df = pd.read_csv('../data/indian_cars_dataset.csv')
    initial_count = len(df)
    print(f"Original row count: {initial_count}")
    
    # 2. Remove exact duplicates and report
    duplicates = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    final_count = len(df)
    print(f"Duplicate rows removed: {duplicates}")
    print(f"Row count after removing duplicates: {final_count}")
    
    # 3. Clean Car_Model
    # Remove literal "['" at the start and "']" at the end if present
    df['Car_Model'] = df['Car_Model'].str.replace(r"^\[\'", "", regex=True)
    df['Car_Model'] = df['Car_Model'].str.replace(r"\'\]$", "", regex=True)
    df['Car_Model'] = df['Car_Model'].str.strip()
    
    # 4. Clean Car_Price
    df['Car_Price'] = df['Car_Price'].astype(str).str.replace(',', '')
    df['Car_Price'] = pd.to_numeric(df['Car_Price'], errors='coerce')
    
    # 5. Clean Car_Kms
    df['Car_Kms'] = df['Car_Kms'].astype(str).str.replace(r' KMs', '', case=False, regex=True)
    df['Car_Kms'] = df['Car_Kms'].str.replace(',', '')
    df['Car_Kms'] = pd.to_numeric(df['Car_Kms'], errors='coerce')
    
    # 6. Clean Car_Location
    df['Car_Location'] = df['Car_Location'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # 7. Preserve original model name (already done in step 3)
    
    # 8. Create Brand column
    df['Brand'] = df['Car_Model'].apply(extract_brand)
    
    # 9. Validate invalid or implausible values
    suspicious = df[
        (df['Car_Year'] < 1950) | (df['Car_Year'] > 2026) |
        (df['Car_Price'] < 10000) | (df['Car_Price'] > 500000000) |
        (df['Car_Kms'] < 0) | (df['Car_Kms'] > 1000000)
    ]
    print(f"Suspicious rows found: {len(suspicious)}")
    if len(suspicious) > 0:
        print("Sample of suspicious rows:")
        print(suspicious[['Car_Model', 'Car_Year', 'Car_Price', 'Car_Kms']].head())
    
    # 10. Check for remaining missing values and duplicates
    missing_values = df.isnull().sum().sum()
    remaining_duplicates = df.duplicated().sum()
    print(f"Missing values after cleaning: {missing_values}")
    print(f"Remaining duplicates: {remaining_duplicates}")
    print("\nFinal column names and data types:")
    print(df.dtypes)
    print("\nPreview of first 5 cleaned rows:")
    print(df.head())
    
    # Save the cleaned dataset
    df.to_csv('../data/cleaned_cars_dataset.csv', index=False)
    print("\nSaved cleaned dataset to data/cleaned_cars_dataset.csv\n")

if __name__ == '__main__':
    main()
