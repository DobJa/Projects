import numpy as np
import tensorflow as tf
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split


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

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001),
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


x_train, x_split, y_train, y_split = train_test_split(data, labels, test_size = 0.3)
x_test, x_val, y_test, y_val = train_test_split(x_split,y_split, test_size = 0.3)

history = model.fit(x_train,y_train, batch_size = 64,
                    epochs = 15,
                    validation_data = (x_val,y_val))

results = model.evaluate(x_test, y_test, batch_size = 32)
