import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation, DataTransformationConfig

@dataclass
class DataIngestion_config():
    train_db_path: str=os.path.join('artifacts', "train.csv")
    test_db_path: str=os.path.join('artifacts', "test.csv")
    raw_db_path: str=os.path.join('artifacts', "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestion_config()

    def initiate_data_ingestion(self):
        logging.info("Initiated the Data Ingestion Method or Component")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info("Reading the dataset")

            os.makedirs(os.path.dirname(self.ingestion_config.train_db_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_db_path,index=False,header=True)

            logging.info("Spilltting the data into train & test")
            train_set, test_set = train_test_split(
                df,test_size=0.2,random_state=42
            )

            train_set.to_csv(
                self.ingestion_config.train_db_path,index=False,header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_db_path,index=False,header=True
            )

            logging.info("Data ingestion successful")

            return(
                self.ingestion_config.train_db_path,
                self.ingestion_config.test_db_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
# if __name__=="__main__":
#     di=DataIngestion()
#     train_data, test_data=di.initiate_data_ingestion()

#     db_transformation=DataTransformation()
#     db_transformation.initiate_data_tranformation(train_data, test_data)