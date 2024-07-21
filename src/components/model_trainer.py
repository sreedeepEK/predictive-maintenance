import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.utils import evaluate_models, save_object
from src.logger import logging
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
    trained_model_filepath: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test array data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestClassifier(),
                "Logistic Regression": LogisticRegression(max_iter=1000),  # Increased max_iter
                "Decision Tree": DecisionTreeClassifier()
            }

            # No hyperparameters for models
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train,
                                                X_test=X_test, y_test=y_test,
                                                models=models)

            # Getting the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            if best_model_score < 0.7:
                raise CustomException("No best model found.")

            logging.info("Model training completed.")

            # Refit the best model on the entire training data
            best_model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_filepath,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            
            print(f"Best model: {best_model_name}")
            print(f"Accuracy of the best model: {accuracy}")
            
        except Exception as e:
            raise CustomException(e, sys)
