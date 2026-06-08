import pandas as pd   
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score

df = pd.read_csv("student_data.csv")
#print(df.head())

x = df.drop("FinalScore",axis=1)
y = df["FinalScore"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(x_train,y_train)

prediction = model.predict(x_test)
mae = mean_absolute_error(y_test,prediction)
r2 = r2_score(y_test,prediction)

print("\nModel Performance")
print("-" * 30)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

with open("model.pkl","wb") as file:
    pickle.dump(model,file)
 
print("Model trained successfully\n")    