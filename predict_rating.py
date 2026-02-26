import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import os

# Configuration
source = r'C:\Users\BALAMURUGAN\OneDrive\Desktop\Data Science\Machine Learning\zoho_dataset\Rotten_Tomatoes_Movies3.xls\Rotten_Tomatoes_Movie.csv'

def load_and_preprocess():
    print(f"Loading data from {source}...")
    # Read as Excel based on previous analysis
    df = pd.read_excel(source, engine='xlrd')
    
    # 1. Drop rows with missing target
    print(f"Initial shape: {df.shape}")
    df = df.dropna(subset=['audience_rating'])
    print(f"Shape after dropping null target: {df.shape}")
    
    # 2. Impute missing numeric values
    df['runtime_in_minutes'] = df['runtime_in_minutes'].fillna(df['runtime_in_minutes'].median())
    
    # 3. Fill missing categorical values
    cat_cols = ['genre', 'directors', 'writers', 'cast', 'studio_name']
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')
        
    # 4. Feature Engineering: Dates
    # Convert dates and extract Year/Month
    df['in_theaters_date'] = pd.to_datetime(df['in_theaters_date'], errors='coerce')
    df['theater_year'] = df['in_theaters_date'].dt.year.fillna(0).astype(int)
    df['theater_month'] = df['in_theaters_date'].dt.month.fillna(0).astype(int)
    
    # 5. Encoding
    # Categorical variables with too many unique values for OneHot
    le = LabelEncoder()
    df['rating_encoded'] = le.fit_transform(df['rating'].astype(str))
    df['tomatometer_status_encoded'] = le.fit_transform(df['tomatometer_status'].astype(str))
    df['studio_encoded'] = le.fit_transform(df['studio_name'].astype(str))
    
    # For Genre, let's take the first genre listed for simplicity in baseline
    df['primary_genre'] = df['genre'].apply(lambda x: x.split(',')[0] if ',' in str(x) else x)
    df['genre_encoded'] = le.fit_transform(df['primary_genre'])
    
    # 6. Select Features
    features = [
        'theater_year', 'theater_month', 'runtime_in_minutes', 
        'tomatometer_rating', 'tomatometer_count', 
        'rating_encoded', 'tomatometer_status_encoded', 
        'studio_encoded', 'genre_encoded'
    ]
    
    X = df[features]
    y = df['audience_rating']
    
    return X, y, features

def train_and_evaluate(X, y, features):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Performance ---")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Save results
    with open('evaluation_results.txt', 'w') as f:
        f.write("--- Model Performance ---\n")
        f.write(f"MAE: {mae:.2f}\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"R2 Score: {r2:.2f}\n\n")
        f.write("--- Feature Importance ---\n")
        importances = model.feature_importances_
        for i, feat in enumerate(features):
            f.write(f"{feat}: {importances[i]:.4f}\n")
            
    print("\nResults saved to evaluation_results.txt")

if __name__ == "__main__":
    try:
        X, y, features = load_and_preprocess()
        train_and_evaluate(X, y, features)
    except Exception as e:
        print(f"Error during execution: {e}")
