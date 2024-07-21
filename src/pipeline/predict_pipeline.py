import os
import sys
import numpy as np
import pandas as pd
from src.utils import load_object
from src.exception import CustomException
from sklearn.preprocessing import OrdinalEncoder

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            
            features_df = pd.DataFrame(features, columns=['Type', 'Air temperature [K]', 'Process temperature [K]', 
                                                        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])
            
            
            if 'Type' in features_df.columns:
                features_df['Type'] = features_df['Type'].map({0: 'L', 1: 'M', 2: 'H'})
            
            
            # Print features for debugging
            print(f"Features DataFrame:\n{features_df}")
            
            
            # Transform the features
            data_scaled = preprocessor.transform(features_df)
            
            # Predict
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, selected_type: str, air_temperature: float, process_temperature: float,
                rotational_speed: float, torque: float, tool_wear: float):
        self.selected_type = selected_type
        self.air_temperature = air_temperature
        self.process_temperature = process_temperature
        self.rotational_speed = rotational_speed
        self.torque = torque
        self.tool_wear = tool_wear
    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "Type": [self.selected_type],
                "Air temperature [K]": [self.air_temperature],
                "Process temperature [K]": [self.process_temperature],
                "Rotational speed [rpm]": [self.rotational_speed],
                "Torque [Nm]": [self.torque],
                "Tool wear [min]": [self.tool_wear]
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
