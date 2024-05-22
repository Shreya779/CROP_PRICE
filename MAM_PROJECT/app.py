from flask import Flask, request, render_template
import pickle
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

app = Flask(__name__)

# Load the trained model, LabelEncoder, feature names, and training data
with open('C:/Users/shrey/PycharmProjects/pythonProject/MAM_PROJECT/model/rf_pipeline.sav', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

with open('X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)

def get_lime_explanation(input_data, model, X_train, feature_names):
    explainer = LimeTabularExplainer(X_train.values, feature_names=feature_names, mode='classification')
    explanation = explainer.explain_instance(input_data[0], model.predict_proba, num_features=len(feature_names))
    return explanation

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = np.array([[float(request.form['nitrogen']),
                            float(request.form['phosphorus']),
                            float(request.form['potassium']),
                            float(request.form['temperature']),
                            float(request.form['humidity']),
                            float(request.form['ph']),
                            float(request.form['rainfall'])]])

    if (input_data <= 0).any():
        return render_template('error.html', message='Input values cannot be zero or negative.')
    prediction = model.predict(input_data)
    crop_name = label_encoder.inverse_transform(prediction)[0]

    lime_explanation = get_lime_explanation(input_data, model, X_train, feature_names)

    return render_template('prediction_result.html',
                           prediction_text=f'Recommended Crop: {crop_name}',
                           explanation_text=lime_explanation.as_html())

if __name__ == '__main__':
    app.run(debug=True)
