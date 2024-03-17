import os
import sys
import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

# this function helps to save the object as file
def save_object(file_path, obj):
    try:
        dir_path=os.path.join(file_path)
        # os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

# this function can help to evaluate the models
    
def evaluate_model(Xtrain, ytrain, Xtest, ytest, models, params):
    try:
        report = {}

        for i in range(len(list(models))):
            model=list(models.values())[i]
            p = params[list(models.keys())[i]]

            gs = GridSearchCV(model,p,cv=3, verbose=1)
            gs.fit(Xtrain,ytrain)

            #train model
            model.set_params(**gs.best_params_)
            model.fit(Xtrain,ytrain) 

            #predicting with trian & test
            train_pred=model.predict(Xtrain)
            test_pred=model.predict(Xtest)

            #gatthering score
            train_set_score=r2_score(ytrain, train_pred)
            test_set_score=r2_score(ytest, test_pred)

            #feeding the scores to dictionary
            report[list(models.keys())[i]]=test_set_score

        return report
    except Exception as e:
        raise CustomException(e,sys)