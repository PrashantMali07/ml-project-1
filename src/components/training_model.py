import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor

from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException

from src.utils import save_object, evaluate_model

#==============Creating class ModelTrainerConfig==============
# this will help to create a path to save the model
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

#Creating the trainer class to train the model & its evalution
class ModelTrainer:
    def __init__(self):
        self.config=ModelTrainerConfig()

    # creating a defination to initiate the training a model
    def initiate_trainer(self, train_arr, test_arr, preprocessor_path, thresold=0.6):
        try:
            logging.info("Splitting the data and feeding it to test & train")

            #spliting it to X & y
            Xtrain, ytrain, Xtest, ytest=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            logging.info("Adding the dictionary for different models")
            # creating a distionary of different models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "K-Neighbours": KNeighborsRegressor(),
                "XGBoost": XGBRFRegressor(),
                "CatBoostReg":CatBoostRegressor(),
                "AdaBoostReg": AdaBoostRegressor(),
                "Linear Regressor": LinearRegression()
            }

            #evaluating the models and getting the reports
            logging.info("Training the models to obtain a best model")
            model_report: dict=evaluate_model(Xtrain=Xtrain, ytrain=ytrain, Xtest=Xtest, ytest=ytest, models=models)

            # obtaining the best model score
            best_model_score=max(sorted(model_report.values()))

            # obtaining the best model name
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            
            #best model
            best_model=models[best_model_name]

            if best_model_score<thresold:
                raise CustomException("ERROR!!! --> We have no best model.")
                logging.info(f"The model & score: {best_model_name}->{best_model_score}. Model isn't found as the Model accuracy is less than {best_model_score*100}%.")

            logging.info(f"We have found a best model: {best_model_name} with accuracy of {best_model_score*100}%")

            save_object(
                file_path=ModelTrainerConfig.trained_model_file_path,
                obj=best_model
            )

            #predicting the test data & score evaluation
            predicted=best_model.predict(Xtest)

            #score
            R2_score=r2_score(ytest, predicted)

            return R2_score
        except Exception as e:
            raise CustomException(e,sys)

