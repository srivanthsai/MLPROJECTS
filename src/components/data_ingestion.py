import os 
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))
from src.exception import customException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation,datatransformationconfig
from src.components.model_trainer import ModeltrainerConfig,Modeltrainer

@dataclass
class DataIngenstionConfig:
    train_data_path: str= os.path.join('artifacts','train.csv')
    test_data_path: str= os.path.join('artifacts','test.csv')
    raw_data_path: str= os.path.join('artifacts','data.csv')

class DataIngenstion:
    def __init__(self):
        self.ingestion_config = DataIngenstionConfig()

    def initiate_data_ingenstion(self):
        logging.info('Enter the data ingestion method:')
        try:
            df=pd.read_csv('notebooks\data\stud.csv')
            logging.info('imported the dataset')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,header=True, index=False)
            logging.info('the dataset will split into training and testing sets')
            train_data,test_data=train_test_split(df,test_size=0.2,random_state=42)
            train_data.to_csv(self.ingestion_config.train_data_path,header=True, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,header=True, index=False)
            logging.info('The dataset has been split into train and test sets and ingestion has completed')
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise customException(e,sys) 
        
if __name__ == "__main__":
    obj=DataIngenstion()
    train_data,test_data=obj.initiate_data_ingenstion()   

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    model_trainer=Modeltrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))