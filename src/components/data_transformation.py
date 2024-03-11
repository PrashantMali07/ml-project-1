import os
import sys 
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_object_filepath=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.DB_TRANSFORM_CONFIG=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function can transform the data by performing OneHotEncoder, Imputer & Standard scaler.
        '''
        try:
            numerical_columns=['writing_score','reading_score']
            categorical_columns=[
                'gender', 
                'race_ethnicity', 
                'parental_level_of_education', 
                'lunch', 
                'test_preparation_course'
            ]

            num_pipeline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            logging.info("Treating Outliers & scaling done for numerical columns.")

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Treating categorical columns & scaling done.")

            logging.info("Initializing the preprocessor")

            preprocessor=ColumnTransformer([
                ("num_pipeline", num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)
            ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_tranformation(self,train_path, test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading train & test data")

            logging.info("Initializing the preproccessing object")

            preproccessing_object=self.get_data_transformer_object()

            target_col="math_score"
            numerical_col=['writing_score','reading_score']

            input_feature_train_df=train_df.drop(columns=[target_col], axis=1)
            target_feature_train_df=train_df[target_col]

            input_feature_test_df=test_df.drop(columns=[target_col], axis=1)
            target_feature_test_df=test_df[target_col]

            logging.info(
                f"Initializing preprocessing object on training & testing dataset."
            )

            input_feature_train_arr=preproccessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr=preproccessing_object.transform(input_feature_test_df)

            #concatinating the array of input & traget data for train and test
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving Preprocessor.")

            logging.info(f"Saving object to --> {self.DB_TRANSFORM_CONFIG.preprocessor_object_filepath}")

            save_object(
                file_path=self.DB_TRANSFORM_CONFIG.preprocessor_object_filepath,
                obj=preproccessing_object
            )

            return (
                train_arr,
                test_arr,
                self.DB_TRANSFORM_CONFIG.preprocessor_object_filepath
            )
        except Exception as e:
            raise CustomException(e, sys)