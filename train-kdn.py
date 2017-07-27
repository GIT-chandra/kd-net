import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,Reshape,Flatten

# loading data
X_train = np.load('mnet10_train.npy')
X_test = np.load('mnet10_test.npy')
labels_train = np.load('mnet10_train_labels.npy')
lables_test = np.load('mnet10_test_labels.npy')

BATCH_SIZE = 111
NUM_CLASSES = 10
EPOCHS = 100


X_train = X_train.reshape((len(X_train),1024,3)) 
X_test = X_test.reshape((len(X_test),1024,3)) 
y_train = keras.utils.to_categorical(labels_train,NUM_CLASSES)
y_test = keras.utils.to_categorical(lables_test,NUM_CLASSES)

model = Sequential()

dense1 = Dense(32,input_shape = (1024,3))
model.add(dense1) 
# print(dense1.output_shape) # you get (None,1024,32), 'None' for the batch
model.add(Reshape((1024,32,1)))

#1
conv1 = Conv2D(32,(2,32),strides = (2,1))
model.add(conv1)	
print(conv1.output_shape)
model.add(Activation('relu'))

#2
conv2 = Conv2D(64,(2,1),strides = (2,1))
model.add(conv2)	
print(conv2.output_shape)
model.add(Activation('relu'))

#3
model.add(Conv2D(64,(2,1),strides = (2,1)))	
model.add(Activation('relu'))

#4
model.add(Conv2D(128,(2,1),strides = (2,1)))	
model.add(Activation('relu'))

#5
model.add(Conv2D(128,(2,1),strides = (2,1)))	
model.add(Activation('relu'))

#6
model.add(Conv2D(256,(2,1),strides = (2,1)))	
model.add(Activation('relu'))

#7
model.add(Conv2D(256,(2,1),strides = (2,1)))	
model.add(Activation('relu'))

#8
model.add(Conv2D(512,(2,1),strides = (2,1)))	
model.add(Activation('relu'))

#9
model.add(Conv2D(512,(2,1),strides = (2,1)))	
model.add(Activation('relu'))

#10
conv10 = Conv2D(128,(2,1),strides = (2,1))
model.add(conv10)	
print(conv10.output_shape)
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(10,activation='softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001,decay = 1e-6)

model.compile(loss = 'categorical_crossentropy',optimizer = opt,metrics=['accuracy'])

model.fit(X_train,y_train,epochs = EPOCHS,batch_size = BATCH_SIZE)
model.save('kd-tree.h5')


score = model.evaluate(X_test,y_test,batch_size = BATCH_SIZE)