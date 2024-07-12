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
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "KNeighbors": {
                    'n_neighbors' : [5,7,9,11]
                },
                "XGBoost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Classifier":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }
            model_report:dict = evaluate_model(x_train=x_train,x_test=x_test,y_test=y_test,y_train=y_train,models=models,parameters=params)
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