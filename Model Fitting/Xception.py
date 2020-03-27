import pandas as pd
import tensorflow
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import * 
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.applications.xception import Xception

base_model = Xception(weights='imagenet', include_top=False,input_shape = (224,224,3))
x = base_model.output

x = Dense(64,activation = 'relu',bias_regularizer = keras.regularizer.l1(0.1))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Flatten()(x)

predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layers in model.layers[0:123]:
  layers.trainable = False
  
model.compile(optimizer=tensorflow.keras.optimizers.Adagrad(lr = 0.0001,decay = 0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(features,laabels,epochs = 20,batch_size = 64,validation_split=0.2)
