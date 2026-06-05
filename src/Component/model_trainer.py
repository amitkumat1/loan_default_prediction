import os
import sys

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {   
                      'Logistic Regression' : LogisticRegression(),
                      'Decision Tree' : DecisionTreeClassifier(),
                      'Naive Bayes' : GaussianNB(),
                      'Random Forest' : RandomForestClassifier(),
                      'KNN' : KNeighborsClassifier()
                    }
            
            params={
                     'Logistic Regression': {
                         'C': [0.1, 1, 10],
                         'penalty': ['l2'],
                         'solver': ['liblinear'],
                         'max_iter': [100]
                     },

                     'Decision Tree': {
                         'criterion': ['gini'],
                         'max_depth': [None, 10],
                         'min_samples_split': [2, 5],
                         'min_samples_leaf': [1, 2]
                     },

                     'Random Forest': {
                         'n_estimators': [100, 200],
                         'max_depth': [None, 10],
                         'min_samples_split': [2, 5],
                         'min_samples_leaf': [1, 2]
                     },

                     'Naive Bayes': {
                         'var_smoothing': [1e-09, 1e-07]
                     },

                     'KNN': {
                         'n_neighbors': [3, 5, 7],
                         'weights': ['uniform'],
                         'metric': ['euclidean', 'manhattan']
                     }
                    }

            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            score = accuracy_score(y_test, predicted)
            logging.info(f"Best model found on both training and testing dataset")

            return score
        
        except Exception as e:
            raise CustomException(e, sys)