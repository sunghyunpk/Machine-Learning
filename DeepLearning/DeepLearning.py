from keras.datasets import cifar10
(x_train,y_train), (x_test,y_test) = cifar10.load_data()

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

# x_train shape: (50000, 32, 32, 3)
# y_train shape: (50000, 1)
# x_test shape: (10000, 32, 32, 3)
# y_test shape: (10000, 1)

from keras.utils import to_categorical 
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

print(y_train_one_hot)

# [[0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 1.]
#  [0. 0. 0. ... 0. 0. 1.]
#  ...
#  [0. 0. 0. ... 0. 0. 1.]
#  [0. 1. 0. ... 0. 0. 0.]
#  [0. 1. 0. ... 0. 0. 0.]]

x_train = x_train/255 
x_test = x_test/255

#Base Model Implementaion Model
from keras.models import Sequential 
from keras.layers import Dense, Flatten 

Base_model = Sequential()
Base_model.add(Dense(50, activation = 'relu', input_shape = x_train.shape[1:]))
Base_model.add(Flatten())
Base_model.add(Dense(10, activation = 'softmax'))

Base_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

Base_model.fit(x_train, y_train_one_hot,  epochs = 10, batch_size = 256)


# Epoch 1/10
# 50000/50000 [==============================] - 26s 514us/step - loss: 1.7873 - accuracy: 0.3789
# Epoch 2/10
# 50000/50000 [==============================] - 25s 509us/step - loss: 1.5529 - accuracy: 0.4605
# Epoch 3/10
# 50000/50000 [==============================] - 25s 499us/step - loss: 1.4954 - accuracy: 0.4819
# Epoch 4/10
# 50000/50000 [==============================] - 25s 508us/step - loss: 1.4581 - accuracy: 0.4966
# Epoch 5/10
# 50000/50000 [==============================] - 25s 506us/step - loss: 1.4314 - accuracy: 0.5050
# Epoch 6/10
# 50000/50000 [==============================] - 25s 506us/step - loss: 1.4164 - accuracy: 0.5086
# Epoch 7/10
# 50000/50000 [==============================] - 25s 508us/step - loss: 1.3939 - accuracy: 0.5193
# Epoch 8/10
# 50000/50000 [==============================] - 25s 501us/step - loss: 1.3818 - accuracy: 0.5253
# Epoch 9/10
# 50000/50000 [==============================] - 25s 503us/step - loss: 1.3668 - accuracy: 0.5255
# Epoch 10/10
# 50000/50000 [==============================] - 25s 509us/step - loss: 1.3533 - accuracy: 0.5360

accuracy = Base_model.evaluate(x_train, y_train_one_hot)
print(accuracy)

# 50000/50000 [==============================] - 14s 273us/step
# [1.3291387002563477, 0.5426599979400635]

print(Base_model.metrics_names)
# ['loss', 'accuracy']


#First Implementation Model
from keras.models import Sequential 
from keras.layers import Dense, Flatten, Conv2D

First_model = Sequential()
First_model.add (Conv2D(64, (5,5), activation = 'relu',  input_shape = x_train.shape[1:]))
First_model.add (Conv2D(32, (5,5), activation = 'relu'))
First_model.add(Dense(50, activation = 'relu'))
First_model.add(Flatten())
First_model.add(Dense(10, activation = 'softmax'))

First_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

First_model.fit(x_train, y_train_one_hot,  epochs = 10, batch_size = 256)

# Epoch 1/10
# 50000/50000 [==============================] - 193s 4ms/step - loss: 1.6003 - accuracy: 0.4260
# Epoch 2/10
# 50000/50000 [==============================] - 192s 4ms/step - loss: 1.2789 - accuracy: 0.5483
# Epoch 3/10
# 50000/50000 [==============================] - 194s 4ms/step - loss: 1.1377 - accuracy: 0.6026
# Epoch 4/10
# 50000/50000 [==============================] - 194s 4ms/step - loss: 1.0441 - accuracy: 0.6363
# Epoch 5/10
# 50000/50000 [==============================] - 193s 4ms/step - loss: 0.9558 - accuracy: 0.6681
# Epoch 6/10
# 50000/50000 [==============================] - 197s 4ms/step - loss: 0.8955 - accuracy: 0.6885
# Epoch 7/10
# 50000/50000 [==============================] - 193s 4ms/step - loss: 0.8372 - accuracy: 0.7111
# Epoch 8/10
# 50000/50000 [==============================] - 192s 4ms/step - loss: 0.7797 - accuracy: 0.7318
# Epoch 9/10
# 50000/50000 [==============================] - 193s 4ms/step - loss: 0.7238 - accuracy: 0.7500
# Epoch 10/10
# 50000/50000 [==============================] - 194s 4ms/step - loss: 0.6701 - accuracy: 0.7691

accuracy = First_model.evaluate(x_train, y_train_one_hot)
print(accuracy)

# 50000/50000 [==============================] - 54s 1ms/step
# [0.5876765490627289, 0.8046799898147583]


#Second Implementation Model 
from keras.models import Sequential 
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D


Second_model = Sequential()
Second_model.add (Conv2D(64, (5,5), activation = 'relu', input_shape = (32,32,3)))
Second_model.add(MaxPooling2D (pool_size = (2,2)))
Second_model.add(Conv2D(32, (5,5), activation = 'relu'))
Second_model.add(Dense(50, activation = 'relu'))
Second_model.add(Flatten())
Second_model.add(Dense(30, activation = 'relu'))
Second_model.add(Dense(10, activation = 'softmax'))

Second_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

Second_model.fit(x_train, y_train_one_hot,  epochs = 10, batch_size = 256)

# Epoch 1/10
# 50000/50000 [==============================] - 70s 1ms/step - loss: 1.7366 - accuracy: 0.3726
# Epoch 2/10
# 50000/50000 [==============================] - 70s 1ms/step - loss: 1.3741 - accuracy: 0.5095
# Epoch 3/10
# 50000/50000 [==============================] - 69s 1ms/step - loss: 1.2617 - accuracy: 0.5538
# Epoch 4/10
# 50000/50000 [==============================] - 70s 1ms/step - loss: 1.1892 - accuracy: 0.5826
# Epoch 5/10
# 50000/50000 [==============================] - 69s 1ms/step - loss: 1.1280 - accuracy: 0.6041
# Epoch 6/10
# 50000/50000 [==============================] - 69s 1ms/step - loss: 1.0835 - accuracy: 0.6200
# Epoch 7/10
# 50000/50000 [==============================] - 68s 1ms/step - loss: 1.0375 - accuracy: 0.6366
# Epoch 8/10
# 50000/50000 [==============================] - 69s 1ms/step - loss: 1.0000 - accuracy: 0.6517
# Epoch 9/10
# 50000/50000 [==============================] - 70s 1ms/step - loss: 0.9649 - accuracy: 0.6635
# Epoch 10/10
# 50000/50000 [==============================] - 69s 1ms/step - loss: 0.9261 - accuracy: 0.6754

accuracy = Second_model.evaluate(x_train, y_train_one_hot)
print(accuracy)

# 50000/50000 [==============================] - 24s 485us/step
# [0.8872799510765076, 0.6904799938201904]

