import pandas as pd
import pickle
import numpy as np     

with open("model.pkl","rb") as file:
    model = pickle.load(file)   
    
print("Student Performance Predictor\n")
hr = float(input("Enter the study hours: "))
at = float(input("Enter the % attendance: "))   
sc = float(input("Enter the previous score: "))
a = float(input("Enter assignment completed: "))

data = pd.DataFrame({
    "StudyHours": [hr],
    "Attendance": [at],
    "PreviousScore": [sc],
    "AssignmentsCompleted": [a]
})
score = model.predict(data)
print("Predicted Score: ",round(score[0],2))