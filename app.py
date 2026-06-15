from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.utils import load_object

application=Flask(__name__, template_folder='template')

app=application

## Route for a home page

def _safe_cast(val, to_type, default):
    try:
        if val is None or str(val).strip() == "":
            return default
        return to_type(val)
    except Exception:
        return default

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            MaritalStatus=request.form.get('MaritalStatus'),
            EmploymentType=request.form.get('EmploymentType'),
            Education=request.form.get('Education'),
            HasMortgage=request.form.get('HasMortgage'),
            HasDependents=request.form.get('HasDependents'),
            HasCoSigner=request.form.get('HasCoSigner'),
            Age=_safe_cast(request.form.get('Age'), int, 0),
            Income=_safe_cast(request.form.get('Income'), float, 0.0),
            NumCreditLines=_safe_cast(request.form.get('NumCreditLines'), int, 0),
            DTIRatio=_safe_cast(request.form.get('DTIRatio'), float, 0.0),
            LoanTerm=_safe_cast(request.form.get('LoanTerm'), int, 0),
            MonthsEmployed=_safe_cast(request.form.get('MonthsEmployed'), int, 0),
            LoanPurpose=request.form.get('LoanPurpose'),
            InterestRate=_safe_cast(request.form.get('InterestRate'), float, 0.0),
            LoanAmount=_safe_cast(request.form.get('LoanAmount'), float, 0.0),
            CreditScore=_safe_cast(request.form.get('CreditScore'), int, 0),
        )

        
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        
        preds, proba = predict_pipeline.predict(pred_df)
        print("after Prediction")
        # Map numeric prediction to human-readable label
        pred_label = 'Default' if float(preds[0]) == 1.0 else 'No Default'
        prob_val = None
        if proba is not None:
            try:
                prob_val = float(proba[0][1])
            except Exception:
                prob_val = None
        prob_display = f"{prob_val:.2f}" if prob_val is not None else None
        return render_template('home.html', results=pred_label, probability=prob_display)
       
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)   