import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os

class MoviePredictor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.model = None
        self.metadata = {}
        self.features = [
            'theater_year', 'runtime_in_minutes', 
            'tomatometer_rating', 'tomatometer_count', 
            'rating_encoded', 'tomatometer_status_encoded', 
            'studio_encoded', 'genre_encoded'
        ]

    def train(self):
        # Load and preprocess
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Dataset not found at {self.csv_path}")
            
        # df = pd.read_csv(self.csv_path) # Removed to avoid encoding error before try block
        # Checking app.py again: source = ...Rotten_Tomatoes_Movie.csv, but then pd.read_excel(source, engine='xlrd')
        # This is likely a mistake in app.py if it's a CSV. I'll use read_csv or read_excel based on actual check.
        
        # Let's try read_csv with common encodings first
        try:
            df = pd.read_csv(self.csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(self.csv_path, encoding='latin1')
            except Exception:
                # Fallback to excel if reading as CSV fails (in case it's actually an XLS/XLSX renamed to CSV)
                try:
                    df = pd.read_excel(self.csv_path, engine='xlrd')
                except Exception as e:
                    raise Exception(f"Failed to read dataset: {e}")

        df = df.dropna(subset=['audience_rating'])
        
        # Impute
        df['runtime_in_minutes'] = df['runtime_in_minutes'].fillna(df['runtime_in_minutes'].median())
        for col in ['genre', 'directors', 'writers', 'cast', 'studio_name']:
            df[col] = df[col].fillna('Unknown')
        
        df['in_theaters_date'] = pd.to_datetime(df['in_theaters_date'], errors='coerce')
        df['theater_year'] = df['in_theaters_date'].dt.year.fillna(0).astype(int)
        
        # Encoders
        df['primary_genre'] = df['genre'].apply(lambda x: str(x).split(',')[0] if ',' in str(x) else str(x))

        for col in ['rating', 'tomatometer_status', 'studio_name', 'primary_genre']:
            df[col] = df[col].astype(str)

        le_rating = LabelEncoder().fit(df['rating'])
        le_status = LabelEncoder().fit(df['tomatometer_status'])
        le_studio = LabelEncoder().fit(df['studio_name'])
        le_genre = LabelEncoder().fit(df['primary_genre'])
        
        # Encode for training
        df['rating_encoded'] = le_rating.transform(df['rating'])
        df['tomatometer_status_encoded'] = le_status.transform(df['tomatometer_status'])
        df['studio_encoded'] = le_studio.transform(df['studio_name'])
        df['genre_encoded'] = le_genre.transform(df['primary_genre'])
        
        X = df[self.features]
        y = df['audience_rating']
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        # Metadata for UI
        self.metadata = {
            'ratings': sorted(df['rating'].unique().tolist()),
            'statuses': sorted(df['tomatometer_status'].unique().tolist()),
            'genres': sorted(df['primary_genre'].unique().tolist()),
            'studios': sorted(df['studio_name'].unique().tolist()),
            'le_rating': le_rating,
            'le_status': le_status,
            'le_studio': le_studio,
            'le_genre': le_genre
        }
        
    def predict(self, input_dict):
        if self.model is None:
            self.train()
            
        # Prepare input data
        input_data = pd.DataFrame([{
            'theater_year': input_dict['theater_year'],
            'runtime_in_minutes': input_dict['runtime_in_minutes'],
            'tomatometer_rating': input_dict['tomatometer_rating'],
            'tomatometer_count': input_dict['tomatometer_count'],
            'rating_encoded': self.metadata['le_rating'].transform([input_dict['rating']])[0],
            'tomatometer_status_encoded': self.metadata['le_status'].transform([input_dict['status']])[0],
            'studio_encoded': self.metadata['le_studio'].transform([input_dict['studio']])[0],
            'genre_encoded': self.metadata['le_genre'].transform([input_dict['genre']])[0]
        }])
        
        prediction = self.model.predict(input_data)[0]
        return float(prediction)

    def get_ui_metadata(self):
        if not self.metadata:
            self.train()
        return {
            'ratings': self.metadata['ratings'],
            'statuses': self.metadata['statuses'],
            'genres': self.metadata['genres'],
            'studios': self.metadata['studios']
        }
