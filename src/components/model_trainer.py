import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))
from src.exception import customException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModeltrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class Modeltrainer:
    def __init__(self):
        self.model_trainer_config=ModeltrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('splitting the train and test input data')
            x_train, y_train, x_test, y_test=(
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )
            models={
                'Linear Regression': LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'Gradient Boosting': GradientBoostingClassifier(),
                'AdaBoost': AdaBoostRegressor(),
                'KNeighbors': KNeighborsRegressor(),
                'XGBoost': XGBRegressor(),
                'CatBoosting Classifier': CatBoostRegressor()
            }
            
            model_report:dict = evaluate_model(x_train=x_train,x_test=x_test,y_test=y_test,y_train=y_train,models=models)
            best_model_name,best_model_score=max(model_report.items(),key=lambda x: x[1])
            best_model=models[best_model_name]

            if best_model_score <0.6:
                raise customException('No best model has been found')
            logging.info(f'Best model on train and test data: {best_model_name}')

            save_object(self.model_trainer_config.trained_model_file_path,best_model)
            logging.info('best model has been saved to artifacts')

            return best_model_score 
        except Exception as e:
            raise customException(e,sys)