import numpy as np import pandas as pd import matplotlib.pyplot as plt
 
data = pd.read_csv("C:\\Users\\pc\\Desktop\\Sunghyun\\2020\\B455\\dataset\\final_Data.csv")
 
X = data
 
output = data['Deaths']
 
print(output)
 
X.drop(['Name'], axis=1, inplace=True) X.drop(['Deaths'], axis=1, inplace=True) X.drop(['Unnamed: 14'], axis=1, inplace=True) X.drop(['Unnamed: 15'], axis=1, inplace=True) X.drop(['Unnamed: 16'], axis=1, inplace=True) X.drop(['Unnamed: 17'], axis=1, inplace=True) X.drop(['Unnamed: 18'], axis=1, inplace=True) X.drop(['Unnamed: 19'], axis=1, inplace=True) X.drop(['Unnamed: 20'], axis=1, inplace=True) X.drop(['Unnamed: 21'], axis=1, inplace=True) print(X)
 
X= np.asarray(X) output= np.asarray(output)

from sklearn.preprocessing import StandardScaler
 
std_scaler = StandardScaler() std_scaler.fit(X)
X = std_scaler.transform(X) 
X_preprocessed = pd.DataFrame(X, columns = data.columns, index=list(data.index.values)) 
print(X_preprocessed)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, output, test_size=0.3) 
print(y_test)

from sklearn.linear_model import LinearRegression from sklearn.metrics import accuracy_score
 
#Linear Regression 
linreg = LinearRegression() 
linreg.fit(X_train, y_train) 
y_predict = linreg.predict(X_test) 
linreg_score = (linreg.score(X_test, y_test))*100 
print(linreg_score)
# 51.14150014184637 
 
plt.plot(y_predict,label="Predicted Data") 
plt.plot(y_test, label="Actual Data") 
plt.title('Linear Regression') 
plt.ylabel('Acceptance rate') 
plt.legend(loc="lower left") 
plt.show()

#Decision Trees 
from sklearn.tree import DecisionTreeRegressor
decision_Tree = DecisionTreeRegressor(random_state=0) 
decision_Tree.fit(X_train, y_train) 
y_predict = decision_Tree.predict(X_test) 
decision_Tree_score = (decision_Tree.score(X_test, y_test))*100 
print(decision_Tree_score)
# 42.91069459757442

plt.plot(y_predict,label="Predicted Data") 
plt.plot(y_test, label="Actual Data") 
plt.title('Decision Tree') 
plt.ylabel('Acceptance rate') 
plt.legend(loc="lower left") 
plt.show()

#Random Forests 
from sklearn.ensemble import RandomForestRegressor 
ran_Forest = RandomForestRegressor(n_estimators = 100, max_depth = 10, random_state = 0) 
ran_Forest.fit(X_train, y_train) 
y_predict = ran_Forest.predict(X_test) 
ran_Forest_score = (ran_Forest.score(X_test, y_test))*100 
print(ran_Forest_score)
 
plt.plot(y_predict,label="Predicted Data") 
plt.plot(y_test, label="Actual Data") 
plt.title('Random Forest') 
plt.ylabel('Acceptance rate') 
plt.legend(loc="lower left") 
plt.show()
# 80.2736620990994

#Support Vector Machine 
from sklearn.svm import SVC 
from sklearn.svm import SVR
 
svm_accuracy = [] 
svm_accuracy1 = [] 
svm_accuracy2 = [] 
svm_MSE = [] 
svm_MSE1 = [] 
svm_MSE2 = []

#Radial basis function kernel 
svm = SVR(kernel='rbf',gamma='auto') 
svm.fit(X_train,y_train) 
y_predict = svm.predict(X_test) svm_accuracy.append(svm.score(X_test,y_test)*100) 
svm_accuracy = np.asarray(svm_accuracy) 
print(svm_accuracy)
# 12.66369994

#Linear
svm1 = SVR(kernel='linear',gamma='auto') 
svm1.fit(X_train,y_train) 
y_predict1 = svm.predict(X_test) 
svm_accuracy1.append(svm1.score(X_test,y_test)*100) 
svm_accuracy1 = np.asarray(svm_accuracy1) 
print(svm_accuracy1)
# 84.44987584

#Poly
svm2 = SVR(kernel='poly',gamma='auto') 
svm2.fit(X_train,y_train) 
y_predict2 = svm.predict(X_test) 
svm_accuracy2.append(svm2.score(X_test,y_test)*100) 
svm_accuracy2 = np.asarray(svm_accuracy2) 
print(svm_accuracy2)
# 54.00275156

plt.plot(y_predict,label="Predicted Data") 
plt.plot(y_test, label="Actual Data") 
plt.title('SVR rbf') 
plt.ylabel('Acceptance rate') 
plt.legend(loc="lower left") 
plt.show()

plt.plot(y_predict1,label="Predicted Data") 
plt.plot(y_test, label="Actual Data") 
plt.title('SVR Linear') 
plt.ylabel('Acceptance rate') 
plt.legend(loc="lower left") 
plt.show()
  
plt.plot(y_predict2,label="Predicted Data") 
plt.plot(y_test, label="Actual Data") 
plt.title('SVR Poly') 
plt.ylabel('Acceptance rate') 
plt.legend(loc="upper right") 
plt.show()
 
 
 
