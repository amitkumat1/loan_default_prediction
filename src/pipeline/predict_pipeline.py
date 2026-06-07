import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            # Fill missing values to avoid None being passed to encoders
            try:
                features = features.fillna('Missing')
            except Exception:
                pass
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
        MaritalStatus: str,
        EmploymentType: str,
        Education: str,
        HasMortgage: str,
        HasDependents: str,
        HasCoSigner: str,
        Age: int,
        Income: float,
        NumCreditLines: int,
        DTIRatio: float,
        LoanTerm: int,
        MonthsEmployed: int,
        LoanPurpose: str,
        InterestRate: float,
        LoanAmount: float,
        CreditScore: int):

        self.MaritalStatus = MaritalStatus

        self.EmploymentType = EmploymentType

        self.Education = Education

        self.HasMortgage = HasMortgage

        self.HasDependents = HasDependents

        self.HasCoSigner = HasCoSigner

        self.LoanPurpose = LoanPurpose

        self.InterestRate = InterestRate

        self.LoanAmount = LoanAmount

        self.CreditScore = CreditScore

        self.Age = Age

        self.Income = Income

        self.NumCreditLines = NumCreditLines

        self.DTIRatio = DTIRatio

        self.LoanTerm = LoanTerm

        self.MonthsEmployed = MonthsEmployed

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.Age],
                "Income": [self.Income],
                "MaritalStatus": [self.MaritalStatus],
                "EmploymentType": [self.EmploymentType],
                "Education": [self.Education],
                "HasMortgage": [self.HasMortgage],
                "HasDependents": [self.HasDependents],
                "HasCoSigner": [self.HasCoSigner],
                "NumCreditLines": [self.NumCreditLines],
                "DTIRatio": [self.DTIRatio],
                "LoanTerm": [self.LoanTerm],
                "MonthsEmployed": [self.MonthsEmployed],
                "LoanPurpose": [self.LoanPurpose],
                "InterestRate": [self.InterestRate],
                "LoanAmount": [self.LoanAmount],
                "CreditScore": [self.CreditScore],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)