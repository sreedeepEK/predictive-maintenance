from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Home route
@app.route('/')
def home():
    return render_template('index.html', prediction_text=None)  # Initialize prediction_text as None

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    type_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    
    # Get data from form
    selected_type = request.form.get('selected_type')
    selected_type = type_mapping[selected_type]
    
    air_temperature = float(request.form.get('air_temperature'))
    process_temperature = float(request.form.get('process_temperature'))
    rotational_speed = float(request.form.get('rotational_speed'))
    torque = float(request.form.get('torque'))
    tool_wear = float(request.form.get('tool_wear'))

    # Predict
    prediction = model.predict([[selected_type, air_temperature, process_temperature,
                                rotational_speed, torque, tool_wear]])
    
    if prediction[0] == 1:
        result = 'Failure'
    else:
        result = 'No Failure'
    
    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == "__main__":
    app.run(debug=True)
