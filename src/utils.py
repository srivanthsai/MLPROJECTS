import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import customException

def save_object(filepath,obj):
    try:
        dir_path=os.path.dirname(filepath)
        os.makedirs(dir_path,exist_ok=True)
        with open(filepath, 'wb') as file:
            dill.dump(obj, file)
            
    except Exception as e:
        raise customException(e,sys)