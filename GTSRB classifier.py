import numpy as np
import tensorflow as tf
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib as plt
from keras.preprocessing.image import ImageDataGenerator


DATADIR = "D:/PyCharm/GTSRB/Train"

data_list = []
label_list = []
clsamount = 43

base_model = tf.keras.applications.MobileNetV2(input_shape = (80,80,3),
                                               include_top = False,
                                               weights = "imagenet")

base_model.trainable = False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(43)

augment = ImageDataGenerator(
    rotation_range = 40,
    zoom_range = 0.2,
    shear_range= 0.2,
    vertical_flip= True
)

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

model2 = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

model3 = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model2.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model3.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

for cat in range(clsamount):
    cat_path = os.path.join(DATADIR, str(cat))
    for img in os.listdir(cat_path):
        entry = cv2.imread(cat_path + '/' + str(img))
        entry = entry / 255.0
        entry = cv2.resize(entry, (80,80), interpolation=cv2.INTER_NEAREST)
        data_list.append(entry)
        label_list.append(cat)
data = np.array(data_list)
labels = np.array(label_list)


x_train, x_split, y_train, y_split = train_test_split(data, labels, test_size = 0.2)
x_test, x_val, y_test, y_val = train_test_split(x_split,y_split, test_size = 0.5)

history = model.fit(x_train,y_train, batch_size = 64,
                    epochs = 15,
                    shuffle=True,
                    validation_data = (x_val,y_val))
print(f"\n SPLIT1 TESTING \n")
results = model.evaluate(x_test, y_test, batch_size = 32)

augment.fit(x_train)

history2 = model2.fit(augment.flow(x_train, y_train ,batch_size = 64),
                       epochs = 15,
                        shuffle=True,
                       validation_data = (x_val,y_val))
print(f"\n SPLIT2 TESTING \n")

results2 = model2.evaluate(x_test,y_test, batch_size = 32)

x_train3 = np.concatenate((x_train, x_val))
y_train3 = np.concatenate((y_train, y_val))


history3 = model3.fit(augment.flow(x_train3,y_train3, batch_size= 64),
                                epochs=15,
                                shuffle=True,
                                validation_data=(x_val,y_val))
print(f"\n SPLIT3 TESTING \n")

results3 = model3.evaluate(x_test,y_test, batch_size = 32)

