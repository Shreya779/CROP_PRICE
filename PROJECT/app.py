from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)

# Load the CatBoost model
with open('catboost_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def apply_water_penalty(water_avail, yield_pred):
    lower_threshold = 6000
    upper_threshold = 11000
    if water_avail < lower_threshold:
        water_penalty = np.exp(-(lower_threshold - water_avail) ** 2 / (2 * (950 ** 2)))
    elif water_avail > upper_threshold:
        water_penalty = np.exp(-(water_avail - upper_threshold) ** 2 / (2 * (950 ** 2)))
    else:
        water_penalty = 1.0
    return yield_pred * water_penalty

def adjust_yield_ph(ph_level, yield_pred):
    ph_penalty = np.where(
        (ph_level <= 5) | (ph_level >= 10),
        np.exp(-(ph_level - 7) ** 2 / (2 * (1 ** 2))),
        1.0
    )
    return yield_pred * ph_penalty

def adjust_yield_irrigation(irrigation_method, yield_pred):
    if irrigation_method == 'Drip':
        return yield_pred * 1.1
    elif irrigation_method == 'Sprinkler':
        return yield_pred * 1.0
    elif irrigation_method == 'Canal':
        return yield_pred * 0.95
    elif irrigation_method == 'Tube Well':
        return yield_pred * 0.90
    else:
        return yield_pred

def create_future_features(future_dates):
    future_features = pd.DataFrame({'Date': future_dates})
    future_features['year'] = future_features['Date'].dt.year
    future_features['month'] = future_features['Date'].dt.month
    future_features['day'] = future_features['Date'].dt.day
    future_features.drop(columns=['Date'], inplace=True)
    return future_features

# Load the trained model
with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Function to create features for future dates
def create_future_features(future_dates):
    future_features = pd.DataFrame({'Date': future_dates})
    future_features['year'] = future_features['Date'].dt.year
    future_features['month'] = future_features['Date'].dt.month
    future_features['day'] = future_features['Date'].dt.day
    future_features.drop(columns=['Date'], inplace=True)
    return future_features

# Data structure to store the number of days each crop can be stored
crop_storage_days = {
    "Wheat": 270,
    "Barley": 270,
    "Onion": 180,
    "Bajra": 180,
    "Chilli": 60,
    "Coriander": 30,
    "Citrus": 60,
    "Cotton": 180,
    "Fennel": 180,
    "Fenugreek": 30,
    "Garlic": 150,
    "Gram": 180,
    "Guava": 14,
    "Maize": 180,
    "Mango": 14,
    "Mustard": 180,
    "Oilseeds": 180,
    "Opium": 180,
    "Pulses": 180,
    "Sugarcane": 1,
    "Tomato": 14
}

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
    user_df = pd.DataFrame([data])
    y_user_pred = model.predict(user_df)
    water_avail = float(user_df['Water Availability (liters/hectare)'])
    y_user_pred_with_penalty = apply_water_penalty(water_avail, y_user_pred)
    ph_level = float(user_df['pH Level'])
    y_user_pred_with_penalty = adjust_yield_ph(ph_level, y_user_pred_with_penalty)
    irrigation_method = user_df['Irrigation Method'].iloc[0]
    y_user_pred_with_penalty = adjust_yield_irrigation(irrigation_method, y_user_pred_with_penalty)
    return jsonify({"predicted_yield": y_user_pred_with_penalty[0]})

@app.route('/predict2', methods=['POST'])
def predict2():
    district = request.form.get("District")
    crop = request.form.get("Crop")
    market = request.form.get("Market")
    input_date = request.form.get("Date")
    num_days = int(request.form.get("NumberOfDays"))

    input_date = datetime.strptime(input_date, "%Y-%m-%d")
    future_dates = [input_date + timedelta(days=i) for i in range(1, num_days + 1)]

    future_features = create_future_features(future_dates)
    future_predictions_rf = rf_model.predict(future_features)

    max_price_index = future_predictions_rf.argmax()
    max_price_date = future_dates[max_price_index].strftime('%Y-%m-%d')
    max_price = future_predictions_rf[max_price_index]

    predictions = [{"Date": date.strftime('%Y-%m-%d'), "Predicted_Price": price} for date, price in zip(future_dates, future_predictions_rf)]

    # Check if max price date is within the storage period for the selected crop
    if crop in crop_storage_days:
        storage_days = crop_storage_days[crop]
        if datetime.strptime(max_price_date, "%Y-%m-%d") <= input_date + timedelta(days=storage_days):
            max_profit_date = (input_date + timedelta(days=storage_days)).strftime('%Y-%m-%d')
            max_profit_message = f"On {max_price_date}, you can sell your {crop} to maximize profit."
        else:
            max_profit_message = f"The max price date falls outside the storage period for {crop}."
    else:
        max_profit_message = "No information available for storage period of the selected crop."

    return render_template('storage.html', predictions=predictions, max_price=max_price, max_price_date=max_price_date, num_days=num_days, max_profit_message=max_profit_message)

@app.route('/predict1', methods=['POST'])
def predict1():
    district = request.form.get("District")
    crop = request.form.get("Crop")
    market = request.form.get("Market")
    input_date = request.form.get("Date")
    num_days = int(request.form.get("NumberOfDays"))

    input_date = datetime.strptime(input_date, "%Y-%m-%d")
    future_dates = [input_date + timedelta(days=i) for i in range(1, num_days + 1)]

    future_features = create_future_features(future_dates)
    future_predictions_rf = rf_model.predict(future_features)

    max_price_index = future_predictions_rf.argmax()
    max_price_date = future_dates[max_price_index].strftime('%Y-%m-%d')
    max_price = future_predictions_rf[max_price_index]

    predictions = [{"Date": date.strftime('%Y-%m-%d'), "Predicted_Price": price} for date, price in zip(future_dates, future_predictions_rf)]

    return render_template('price.html', predictions=predictions, max_price=max_price, max_price_date=max_price_date, num_days=num_days)

if __name__ == '__main__':
    app.run(debug=True)