import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os

# Set page config
st.set_page_config(page_title="Rotten Tomatoes Predictor", page_icon="🎬", layout="centered")

st.title("🎬 Movie Audience Rating Predictor")
st.markdown("""
Predict how much audiences will like a movie based on critical reception, genre, and more!
""")

# Path to data
source = r'C:\Users\BALAMURUGAN\OneDrive\Desktop\Data Science\Machine Learning\zoho_dataset\Rotten_Tomatoes_Movies3.xls\Rotten_Tomatoes_Movie.csv'

@st.cache_resource
def get_model_and_data():
    # Load and preprocess
    df = pd.read_excel(source, engine='xlrd')
    df = df.dropna(subset=['audience_rating'])
    
    # Impute
    df['runtime_in_minutes'] = df['runtime_in_minutes'].fillna(df['runtime_in_minutes'].median())
    for col in ['genre', 'directors', 'writers', 'cast', 'studio_name']:
        df[col] = df[col].fillna('Unknown')
    
    df['in_theaters_date'] = pd.to_datetime(df['in_theaters_date'], errors='coerce')
    df['theater_year'] = df['in_theaters_date'].dt.year.fillna(0).astype(int)
    
    # Encoders
    # Create primary_genre before casting
    df['primary_genre'] = df['genre'].apply(lambda x: x.split(',')[0] if ',' in str(x) else x)

    # Ensure all values are strings to avoid sorting/comparison errors
    df['rating'] = df['rating'].astype(str)
    df['tomatometer_status'] = df['tomatometer_status'].astype(str)
    df['studio_name'] = df['studio_name'].astype(str)
    df['primary_genre'] = df['primary_genre'].astype(str)

    le_rating = LabelEncoder().fit(df['rating'])
    le_status = LabelEncoder().fit(df['tomatometer_status'])
    le_studio = LabelEncoder().fit(df['studio_name'])
    le_genre = LabelEncoder().fit(df['primary_genre'])
    
    # Encode for training
    df['rating_encoded'] = le_rating.transform(df['rating'])
    df['tomatometer_status_encoded'] = le_status.transform(df['tomatometer_status'])
    df['studio_encoded'] = le_studio.transform(df['studio_name'])
    df['genre_encoded'] = le_genre.transform(df['primary_genre'])
    
    features = [
        'theater_year', 'runtime_in_minutes', 
        'tomatometer_rating', 'tomatometer_count', 
        'rating_encoded', 'tomatometer_status_encoded', 
        'studio_encoded', 'genre_encoded'
    ]
    
    X = df[features]
    y = df['audience_rating']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Metadata for UI
    metadata = {
        'ratings': sorted(df['rating'].unique().tolist()),
        'statuses': sorted(df['tomatometer_status'].unique().tolist()),
        'genres': sorted(df['primary_genre'].unique().tolist()),
        'studios': sorted(df['studio_name'].unique().tolist()),
        'le_rating': le_rating,
        'le_status': le_status,
        'le_studio': le_studio,
        'le_genre': le_genre
    }
    
    return model, metadata

try:
    with st.spinner("Loading AI model..."):
        model, meta = get_model_and_data()

    # Sidebar Inputs
    st.sidebar.header("Movie Details")
    
    input_year = st.sidebar.number_input("Release Year", min_value=1900, max_value=2026, value=2024)
    input_runtime = st.sidebar.number_input("Runtime (Minutes)", min_value=1, max_value=500, value=120)
    input_tomato_rating = st.sidebar.slider("Tomatometer Rating (%)", 0, 100, 75)
    input_tomato_count = st.sidebar.number_input("Critics Count", min_value=1, max_value=1000, value=150)
    
    input_rating = st.sidebar.selectbox("Parental Rating", meta['ratings'])
    input_status = st.sidebar.selectbox("Tomatometer Status", meta['statuses'])
    input_genre = st.sidebar.selectbox("Primary Genre", meta['genres'])
    input_studio = st.sidebar.selectbox("Studio", meta['studios'], index=meta['studios'].index('Universal Pictures') if 'Universal Pictures' in meta['studios'] else 0)

    # Prepare input data
    # Handle unseen labels for encoders by defaulting to nearest or most frequent if needed, 
    # but here we use selects from the meta so they will be seen.
    input_data = pd.DataFrame([{
        'theater_year': input_year,
        'runtime_in_minutes': input_runtime,
        'tomatometer_rating': input_tomato_rating,
        'tomatometer_count': input_tomato_count,
        'rating_encoded': meta['le_rating'].transform([input_rating])[0],
        'tomatometer_status_encoded': meta['le_status'].transform([input_status])[0],
        'studio_encoded': meta['le_studio'].transform([input_studio])[0],
        'genre_encoded': meta['le_genre'].transform([input_genre])[0]
    }])

    # Prediction
    if st.button("Predict Audience Rating"):
        prediction = model.predict(input_data)[0]
        
        # Display Result
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicted Audience Score", f"{prediction:.1f}%")
            
        with col2:
            sentiment = "Highly Positive" if prediction > 80 else "Positive" if prediction > 60 else "Mixed" if prediction > 40 else "Negative"
            st.metric("Predicted Sentiment", sentiment)
            
        # Comparison with Tomato Rating
        diff = prediction - input_tomato_rating
        if diff > 10:
            st.info(f"Audience is expected to be more enthusiastic than critics by {diff:.1f} points!")
        elif diff < -10:
            st.warning(f"Audience might be more critical than reviewers by {abs(diff):.1f} points.")
        else:
            st.success("Audience and Critics are likely to agree on this one!")

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.info("Please ensure the dataset file is accessible.")
