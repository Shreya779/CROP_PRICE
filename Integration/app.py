from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle

app = Flask(__name__)

# Load the Random Forest model
with open('rf_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

# Load the crop-market combinations
with open("crop_market_combinations.pkl", "rb") as f:
    crop_market_combinations = pickle.load(f)

# Function to create features for future dates
def create_future_features(future_dates, crop_market_combinations):
    future_features = pd.DataFrame({'Date': future_dates})
    future_features['year'] = future_features['Date'].dt.year
    future_features['month'] = future_features['Date'].dt.month
    future_features['day'] = future_features['Date'].dt.day

    # Create columns for each crop and market combination
    for _, row in crop_market_combinations.iterrows():
        crop = row['Crop']
        market = row['Market']
        if f'Crop_{crop}' in future_features.columns:
            future_features[f'Crop_{crop}'] = 0
        if f'Market_{market}' in future_features.columns:
            future_features[f'Market_{market}'] = 0

    # Drop the "Date" column
    future_features.drop(columns=['Date'], inplace=True)

    return future_features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/yield')
def yield_page():
    return render_template('yield.html')

@app.route('/price')
def price_page():
    return render_template('price.html')

@app.route('/storage')
def storage_page():
    return render_template('storage.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Process the data and make predictions
    # Replace this with your prediction logic
    return jsonify({"predicted_yield": 1000})  # Dummy prediction for demonstration

@app.route('/predict2', methods=['POST'])
def predict2():
    district = request.form.get("District")
    crop = request.form.get("Crop")
    market = request.form.get("Market")
    input_date = request.form.get("Date")
    num_days = int(request.form.get("NumberOfDays"))

    input_date = datetime.strptime(input_date, "%Y-%m-%d")
    future_dates = [input_date + timedelta(days=i) for i in range(1, num_days + 1)]

    future_features = create_future_features(future_dates, crop_market_combinations)
    future_predictions_rf = rf_model.predict(future_features)

    # Example code for getting max price date
    max_price_index = future_predictions_rf.argmax()
    max_price_date = future_dates[max_price_index].strftime('%Y-%m-%d')
    max_price = future_predictions_rf[max_price_index]

    predictions = [{"Date": date.strftime('%Y-%m-%d'), "Predicted_Price": price} for date, price in zip(future_dates, future_predictions_rf)]

    return jsonify({
        "predictions": predictions,
        "max_price": max_price,
        "max_price_date": max_price_date,
        "num_days": num_days
    })

if __name__ == '__main__':
    app.run(debug=True)
