from flask import Flask, request, jsonify
from flask_cors import CORS
from predictor import MoviePredictor
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize predictor
CSV_PATH = r'C:\Users\BALAMURUGAN\OneDrive\Desktop\Data Science\Machine Learning\zoho_dataset\Rotten_Tomatoes_Movies3.xls\Rotten_Tomatoes_Movie.csv'
predictor = MoviePredictor(CSV_PATH)

@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    try:
        metadata = predictor.get_ui_metadata()
        return jsonify(metadata)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Validate data
        required_fields = [
            'theater_year', 'runtime_in_minutes', 'tomatometer_rating', 
            'tomatometer_count', 'rating', 'status', 'studio', 'genre'
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
        
        prediction = predictor.predict(data)
        
        # Determine sentiment
        sentiment = "Highly Positive" if prediction > 80 else "Positive" if prediction > 60 else "Mixed" if prediction > 40 else "Negative"
        
        return jsonify({
            "prediction": prediction,
            "sentiment": sentiment
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    # Train model on startup if needed
    print("Training model...")
    predictor.train()
    print("Model trained and ready.")
    app.run(debug=True, port=5000)
