import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
import joblib
import json
import os

def main():
    print("=== Training Car Models ===")
    df = pd.read_csv('../data/cleaned_cars_dataset.csv')
    
    categorical_features = ['Car_Model', 'Car_Fuel', 'Car_Location', 'Brand']
    numeric_features = ['Car_Year', 'Car_Kms']
    target = 'Car_Price'
    
    X = df[categorical_features + numeric_features]
    y = df[target]
    
    print(f"Total records: {len(df)}")
    print("Features:", X.columns.tolist())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train records: {len(X_train)}, Test records: {len(X_test)}")
    
    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    models = {
        'Baseline (Median)': DummyRegressor(strategy='median'),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
        'Extra Trees': ExtraTreesRegressor(random_state=42, n_estimators=100)
    }
    
    results = {}
    best_model = None
    best_r2 = -float('inf')
    best_name = ""
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', model)])
        
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        print(f"{name} - MAE: ₹{mae:,.2f}, RMSE: ₹{rmse:,.2f}, R²: {r2:.4f}")
        
        if name != 'Baseline (Median)' and r2 > best_r2:
            best_r2 = r2
            best_model = pipeline
            best_name = name
            
    print(f"\nSelected Model: {best_name} with R²: {best_r2:.4f}")
    
    unique_models = df['Car_Model'].nunique()
    print(f"Data limitation note: High cardinality in Car_Model ({unique_models} unique values) for only {len(df)} records.")
    
    os.makedirs('../models', exist_ok=True)
    joblib.dump(best_model, '../models/car_price_model.joblib')
    
    with open('../models/car_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
        
if __name__ == '__main__':
    main()
