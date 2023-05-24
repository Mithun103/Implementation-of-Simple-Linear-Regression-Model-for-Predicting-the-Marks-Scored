# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries and read the given .csv file into the program
2. Segregate the data into two variables (x-Hours and y-scores)
3. Split the data into Training and Test Data set
4. Import Linear Regression from sklearn display the predicted and the actual values.
5. Plot graph for both Training and Test dataset using matplot library.
6. Finally find the Mean Square Error,Mean absolute Error,Root Mean Square Error.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MITHUN M S
RegisterNumber: 212222240067
*/
```

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
#displaying the content in datafile
df.head()
df.tail()

#segregating data to variables
x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

#splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0) 

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#displaying predicted values
y_pred

#displaying actual values
y_test

#graph plot for training data
plt.scatter(x_train,y_train,color="magenta")
plt.plot(x_train,regressor.predict(x_train),color="teal")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(x_test,y_test,color="magenta")
plt.plot(x_test,regressor.predict(x_test),color="orange")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
## Reading the data set (df.head() & df.tail())
![ml1](https://user-images.githubusercontent.com/93427253/228925523-429137f8-012f-43ef-a5cf-ccc559a484d4.png)
![ml2](https://user-images.githubusercontent.com/93427253/228926161-485f2c42-715d-453b-a730-e6434fcfc8db.png)

## Segregating Data into variables
![ml3](https://user-images.githubusercontent.com/93427253/228926193-9cf14aec-a3e4-47e2-bfd5-c4c544ac1e1e.png)

## Predicted Y values
![ml31](https://user-images.githubusercontent.com/93427253/228925632-05ab55fd-81d9-441f-8f5f-3e8bdde2c8ee.png)
![ml4](https://user-images.githubusercontent.com/93427253/228926249-23a6deba-206a-4418-8743-ed37b7017c79.png)

## Actual Y values
![ml41](https://user-images.githubusercontent.com/93427253/228925792-22e11b9a-869d-403d-b17c-aa64761117fa.png)

## Graph for Training Data
![ml6](https://user-images.githubusercontent.com/93427253/228926326-42b3e4af-1f83-4e87-8a11-741610f276b5.png)

## Graph for Test Data
![ml7](https://user-images.githubusercontent.com/93427253/228926347-7239910d-2e9a-4521-938b-d61d905d357e.png)

## Finding the Errors
![ml8](https://user-images.githubusercontent.com/93427253/228926375-99638820-dcca-4aae-81e0-48b95478592c.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
