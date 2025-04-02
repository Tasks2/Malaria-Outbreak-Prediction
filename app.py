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

        # Categorize input values for contextual explanation
        # Temperature categories
        if temperature < 25:
            temp_category = "low"
        elif temperature < 35:
            temp_category = "moderate"
        else:
            temp_category = "high"
            
        # Humidity categories
        if humidity < 65:
            humidity_category = "low"
        elif humidity < 75:
            humidity_category = "moderate"
        else:
            humidity_category = "high"
            
        # Rainfall categories
        if rainfall < 50:
            rainfall_category = "low"
        elif rainfall < 120:
            rainfall_category = "moderate"
        else:
            rainfall_category = "high"

        # Generate contextual explanation based on input values
        if prediction == 1:
            risk_type = "High risk of malaria"
            
            # Create explanation based on actual input values
            explanation_factors = []
            if temp_category == "high":
                explanation_factors.append("high temperature")
            if humidity_category == "high":
                explanation_factors.append("high humidity")
            if rainfall_category in ["moderate", "high"]:
                explanation_factors.append("significant rainfall")
                
            if explanation_factors:
                factors_text = ", ".join(explanation_factors[:-1])
                if len(explanation_factors) > 1:
                    factors_text += f" and {explanation_factors[-1]}"
                else:
                    factors_text = explanation_factors[0]
                
                explanation = (
                    f"Based on the input parameters, the model estimates a {prob_high:.2f}% chance of high malaria risk. "
                    f"The combination of {factors_text} creates favorable conditions for mosquito breeding and malaria transmission. "
                    f"These environmental factors increase the likelihood of mosquito population growth in the Malindi region."
                )
            else:
                # Fallback for unusual combinations that still predict high risk
                explanation = (
                    f"Based on the input parameters, the model estimates a {prob_high:.2f}% chance of high malaria risk. "
                    f"While individual factors may not be in critical ranges, their specific combination creates conditions "
                    f"that can support mosquito breeding and malaria transmission in the Malindi region."
                )
                
            recommendations = (
                "Environmental conditions indicate high malaria transmission potential in Malindi. Immediate clinical actions recommended:\n"
                "Increase clinical staffing in outpatient departments to manage potential surge in febrile cases\n"
                "Activate rapid response teams for community-based active case detection\n"
                "Ensure adequate stock of antimalarials (ACTs), IV artesunate, and diagnostic supplies\n"
                "Prioritize vulnerable populations (pregnant women, children <5, immunocompromised) for prophylactic interventions\n"
                "Initiate emergency risk communication to healthcare facilities in surrounding regions\n"
                "Establish additional mobile clinics in underserved areas with limited healthcare access\n "
            )
        else:
            risk_type = "Low risk of malaria"
            
            # Create explanation based on actual input values
            protective_factors = []
            if temp_category == "low":
                protective_factors.append("lower temperature")
            if humidity_category == "low":
                protective_factors.append("lower humidity")
            if rainfall_category == "low":
                protective_factors.append("minimal rainfall")
                
            if protective_factors:
                factors_text = ", ".join(protective_factors[:-1])
                if len(protective_factors) > 1:
                    factors_text += f" and {protective_factors[-1]}"
                else:
                    factors_text = protective_factors[0]
                
                explanation = (
                    f"Based on the input parameters, the model estimates a {prob_high:.2f}% chance of high malaria risk. "
                    f"The {factors_text} creates less favorable conditions for mosquito breeding and activity. "
                    f"These environmental factors typically reduce mosquito population growth in the Malindi region."
                )
            else:
                # Fallback for unusual combinations that still predict low risk
                explanation = (
                    f"Based on the input parameters, the model estimates a {prob_high:.2f}% chance of high malaria risk. "
                    f"The specific combination of temperature ({temperature}°C), humidity ({humidity}%), and rainfall ({rainfall}mm) "
                    f"creates conditions that are less conducive to mosquito breeding and malaria transmission in the Malindi region."
                )
                
            recommendations = (
                "Current environmental conditions indicate reduced malaria transmission potential in Malindi. Recommended clinical preparedness:\n"
                "Maintain standard malaria surveillance protocols and case reporting systems\n"
                "Ensure diagnostic supplies (rapid tests, microscopy materials) are within expiration dates\n"
                "Continue routine vector surveillance in known hotspots\n"
                "Maintain baseline antimalarial medication inventory\n"
                "Use this lower-risk period to conduct healthcare staff refresher training on malaria diagnosis and treatment protocols\n"
                "Consider targeted outreach to educate communities on preventive measures before the high-risk season begins\n "
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

