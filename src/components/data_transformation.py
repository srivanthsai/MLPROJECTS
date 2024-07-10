import sys
import os 
from dataclasses import dataclass
import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import customException
from src.logger import logging
from src.utils import save_object

@dataclass
class datatransformationconfig:
    preprocessor_obj_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=datatransformationconfig()
    def get_data_trans_obj(self):
        try:
            #trying to predict math scores
            numerical_features=['writing_score','reading_score']
            categorical_features=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehot',OneHotEncoder(handle_unknown='ignore')),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            logging.info('numerical cols standardization completed: {}'.format(numerical_features))
            logging.info('categorical cols onehot encoding completed: {}'.format(categorical_features))

            preprocessor=ColumnTransformer(
                [
                    ('numerical_pipeline',num_pipeline,numerical_features),
                    ('categorical_pipeline',cat_pipeline,categorical_features)
                ]
            )
            return preprocessor
        
        except  Exception as e:
            raise customException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Data loading completed')

            data_tranformation_obj=self.get_data_trans_obj()

            logging.info('Data preprocessor object initiated')

            target_col='math_score'
            numerical_features=['writing_score','reading_score']

            input_feature_train_df=train_df.drop(columns=[target_col],axis=1)
            target_feature_train_df=train_df[target_col]
            input_feature_test_df=test_df.drop(columns=[target_col],axis=1)
            target_feature_test_df=test_df[target_col]

            logging.info('splitting of the data into target and features is completed')

            logging.info('Transforming the data using preprocessor object')

            input_feature_train_arr= data_tranformation_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=data_tranformation_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            save_object(
                filepath=self.data_transformation_config.preprocessor_obj_path,
                obj=data_tranformation_obj
            )
            
            logging.info('preprocessing object saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path
            )
        except Exception as e:
            raise customException(e,sys)
