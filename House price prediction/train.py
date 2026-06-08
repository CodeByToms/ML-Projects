import pandas as pd   
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # type: ignore
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score

df = pd.read_csv("data.csv")

x = df.drop(['price'],axis=1)
y = df['price']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)
model = LinearRegression()
model.fit(x_train,y_train)

prediction = model.predict(x_test)
mae = mean_absolute_error(y_test,prediction)
r2 = r2_score(y_test,prediction)

print("Model Evaluation\n")
print("Mean absolute error: ",mae)
print("\n")
print("R2 Score: ",r2)

plt.scatter(y_test,prediction)
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Actual price vs Predicted price")
plt.show()