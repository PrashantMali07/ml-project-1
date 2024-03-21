from flask import Flask, request, render_template
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.logger import logging

# Creating the Flask Application
application=Flask(__name__)

app=application

## Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=str(request.form.get('test_preparation_course')),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
    
        getting_data=data.converting_data_to_data_frame()
        print(getting_data)

        pipeline=PredictPipeline()
        results=str(pipeline.PredictionPipeline(getting_data)).replace('[',' ').replace(']',' ')
        logging.info(f"The predicted result is {results} and its dtype is {type(results)}")
        return render_template('home.html',results=results)