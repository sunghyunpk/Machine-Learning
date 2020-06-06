import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
 
data = pd.read_csv("C:\\Users\\pc\\Desktop\\Sunghyun\\2020\\B455\\dataset\\wine.data")
data.tail(50)
data.dropna(inplace = True)
df_label = data[["1"]]
data = data.drop( ["1"], axis = 1)
X = data
print(data)

#One-hat encode 
target = df_label
target.to_numpy()
y = []

for i in np.nditer(target):
    
    if i == 1:
        y.append(np.array([1,0,0]))
    elif i == 2:
        y.append(np.array([0,1,0]))
    else:
        y.append(np.array([0,0,1]))
         

y = np.asarray(y)      
print(y)

#One-hat encode 
target = df_label
target.to_numpy()
y = []

for i in np.nditer(target):
    
    if i == 1:
        y.append(np.array([1,0,0]))
    elif i == 2:
        y.append(np.array([0,1,0]))
    else:
        y.append(np.array([0,0,1]))
         

y = np.asarray(y)      
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3) 



#sigmoid 
def sigmoid(x, derive = False):
    if derive:
        return x * (1-x)
    return 1 / (1 + np.exp(-x)) 

learning_rate = 0.001


w1 = np.random.random((13,20))
w2 = np.random.random((20,3)) 

X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
X_test = (X_test - X_test.min()) / (X_test.max() - X_train.min())

errors = []
#Forward 
epochs = 1000
for i in range(epochs):
    input_layer = X_train 
    hidden_layer = sigmoid(np.dot(input_layer, w1))
    output_layer = sigmoid(np.dot(hidden_layer, w2))

#Backward 
    #Output layer 
    output_layer_error = y_train - output_layer 
    output_layer_delta = output_layer_error * sigmoid(output_layer, derive = True)
        
    #Hidden layer
    hidden_layer_error = output_layer_delta.dot(w2.T)
    hidden_layer_delta = hidden_layer_error * sigmoid(hidden_layer, derive = True)
    
    w1 += input_layer.T.dot(hidden_layer_delta) * learning_rate
    w2 += hidden_layer.T.dot(output_layer_delta) * learning_rate
    

input_layer = X_test
hidden_layer = sigmoid(np.dot(input_layer, w1))
output_layer = sigmoid(np.dot(hidden_layer, w2))
for i in range(len(output_layer)):
    for j in range(len(output_layer[i])):
        if np.amax(output_layer[i]) == output_layer[i,j]:
            output_layer[i,j] = 1
        else:
            output_layer[i,j] = 0  

output_layer_error = y_test - output_layer
error = np.mean(np.abs(output_layer_error))
accuracy = (1 - error) * 100 

print(accuracy)


