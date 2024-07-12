import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import customException

def save_object(filepath,obj):
    try:
        dir_path=os.path.dirname(filepath)
        os.makedirs(dir_path,exist_ok=True)
        with open(filepath, 'wb') as file:
            dill.dump(obj, file)
            
    except Exception as e:
        raise customException(e,sys)

def evaluate_model(x_train,x_test,y_train,y_test,models,parameters):
    try:
        report={}

        for i in range(len(models)):
            model=list(models.values())[i]
            param=parameters[list(models.keys())[i]]
            
            gs=GridSearchCV(model,param,cv=3)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)
            
            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)

            train_model_score=r2_score(y_train,y_train_pred)

            test_model_score=r2_score(y_test,y_test_pred)
            report[list(models.keys())[i]]= test_model_score
        
        return report
    except Exception as e:
        raise customException(e,sys)