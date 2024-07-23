from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

# Initialize the prediction pipeline
pipeline = PredictPipeline()

# Home route
@app.route('/')
def home():
    return render_template('index.html', prediction_text=None)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        type_mapping = {'L': 0, 'M': 1, 'H': 2}
        
        # Get data from form
        selected_type = request.form.get('selected_type')
        if selected_type not in type_mapping:
            raise ValueError("Invalid type selected")

        # Convert selected type to numeric
        selected_type = type_mapping[selected_type]
        
        # Fetch and convert input values
        air_temperature = float(request.form.get('air_temperature'))
        process_temperature = float(request.form.get('process_temperature'))
        rotational_speed = float(request.form.get('rotational_speed'))
        torque = float(request.form.get('torque'))
        tool_wear = float(request.form.get('tool_wear'))

        # Create CustomData instance
        data = CustomData(selected_type, air_temperature, process_temperature,
                            rotational_speed, torque, tool_wear)
        
        # Convert data to DataFrame format for prediction
        df = data.get_data_as_dataframe()
        
        # Predict using the pipeline
        prediction = pipeline.predict(df)
        
        # Interpret prediction
        if prediction[0] == 1:
            result = 'Failure'
            color = 'red'
        else:
            result = 'No Failure'
            color = 'green'
        
        return render_template('index.html', prediction_text=f'Prediction: <span style="color: {color}">{result}</span>')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)
