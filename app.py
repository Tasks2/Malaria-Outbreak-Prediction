import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle

# Create flask app
app = Flask(__name__)

# Load pickle model
model = pickle.load(open('LogisticRegressionMALARIA.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/index')
def predict_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Extract features
        rainfall = float(data['rainfall'])
        humidity = float(data['humidity'])
        temperature = float(data['temperature'])

        # Convert inputs to a NumPy array
        features = np.array([[rainfall, humidity, temperature]])

        # Make prediction and probability score
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        prob_high = proba[1] * 100  # probability percentage for high risk

        # Map prediction to risk type
        if prediction == 1:
            risk_type = "High risk of malaria"
            recommendations = (
                "We recommend installing mosquito nets, using insect repellent, "
                "eliminating standing water around your home, and staying updated on local health alerts. "
                # "Consider visiting a clinic for preventive measures."
                "Preparations for malaria prevention should begin immediately"
            )
        else:
            risk_type = "Low risk of malaria"
            recommendations = (
                "The current conditions suggest a lower risk of malaria. However, it's wise to maintain a clean environment "
                "and stay vigilant to any changes in weather or local outbreak news."
            )

        # Provide an explanation based on the probability
        explanation = (
            f"Based on the input parameters, the model estimates a {prob_high:.2f}% chance of high malaria risk. "
            "High humidity and temperature combined with significant rainfall can create an ideal breeding environment for mosquitoes."
        )

        # Return JSON response with extra details
        return jsonify({
            'prediction': risk_type,
            'probability': f"{prob_high:.2f}%",
            'explanation': explanation,
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
