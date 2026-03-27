# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Load the dataset and remove unnecessary columns like car_ID and CarName.

2.Convert categorical data into numerical form using one-hot encoding.

3.Split the dataset into features (X) and target variable (price), then divide into training and testing sets.

4.Create and train the Linear Regression model using the training data.

5.Perform cross-validation and predict prices using the test data.

6.Evaluate the model using MSE, MAE, R² score and visualize actual vs predicted values.

## Program:
```
/*
Program to implement the multiple linear regression model for predicting car prices with cross-validation.
Developed by: Arunjuthan.M.A
RegisterNumber:  212225230020
*/
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt

data=pd.read_csv("CarPrice_Assignment.csv")
data.head()
data=data.drop(['car_ID','CarName'],axis=1)
data=pd.get_dummies(data,drop_first=True)
data.head()
x=data.drop('price',axis=1)
y=data['price']
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

print("Name : Arunjuthan.M.A")
print("Reg no: 212225230020")
print("\n=== Cross Validation ===")
cv_scores = cross_val_score(model,x,y,cv=5)
print("Fold R_2 scores : ",[f"{score:.4f}" for score in cv_scores])
print(f"Average R_2     : {cv_scores.mean():.4f}")

y_pred=model.predict(x_test)
print("\n=== Test Set Performance ===")
print(f"MSE : {mean_squared_error(y_test,y_pred):.2f}")
print(f"MAE : {mean_absolute_error(y_test,y_pred):.2f}")
print(f"R_2 : {r2_score(y_test,y_pred):.4f}")

plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Prdicted Prices")
plt.grid(True)
plt.show()
```

## Output:

<img width="940" height="762" alt="image" src="https://github.com/user-attachments/assets/41a6c68e-2034-4769-be4f-ecd9faae703d" />


## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
