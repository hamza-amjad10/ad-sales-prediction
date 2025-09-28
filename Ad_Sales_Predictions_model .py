import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error



df=pd.read_csv("6 advertising.csv")

# print(df.head(10))


# Now we train and test and split the data

# first we separate features and target varaible
X=df.drop('Sales',axis=1)
Y=df['Sales']

# now train test and split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

# now check all the shapes and verify data is split train or test?
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# Now we scale the data using standard scaler
standard_sacle_data=StandardScaler()

X_train_scaled=standard_sacle_data.fit_transform(X_train)
X_test_scaled=standard_sacle_data.transform(X_test)


print(X_train_scaled)
print(X_test_scaled)


# now load the linear regression model
linear_regression_model=LinearRegression()


# now train the model
linear_regression_model.fit(X_train_scaled,Y_train)


# now make prediction
Y_pred=linear_regression_model.predict(X_test_scaled)


print(Y_pred)

# r2_score value is:
print("R2 Score: ", r2_score(Y_test,Y_pred))

# mean square error value is:
print("Mean Square Error: ", mean_squared_error(Y_test,Y_pred))



# lets predict sales for this data
new_data = np.array([[230.1, 37.8, 69.2]])

# scale the data 
new_data_scaled = standard_sacle_data.transform(new_data)

# make prediction
new_prediction = linear_regression_model.predict(new_data_scaled)


# print the prediction
print("Predicted Sales:", new_prediction[0])


joblib.dump(linear_regression_model,'ad_sales_model.pkl')
joblib.dump(standard_sacle_data,'ad_sales_scaler.pkl')