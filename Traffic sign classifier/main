import numpy as np
import tensorflow as tf
import os
import cv2
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATADIR = "D:/PyCharm/GTSRB/Train"
DATADIR_AUG = "D:/PyCharm/GTSRB/TrainNorm"

data_list = []
label_list = []
data_list2 = []
label_list2 = []
clsamount = 43

base_model = tf.keras.applications.ResNet101V2(input_shape = (80,80,3),
                                               include_top = False,
                                               weights = "imagenet")
base_model.trainable = False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
disc_layer1 = tf.keras.layers.Dense(1024, activation = "relu")
disc_layer2 = tf.keras.layers.Dense(512, activation = "relu")
prediction_layer = tf.keras.layers.Dense(43, activation = "softmax")


model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    disc_layer1,
    disc_layer2,
    prediction_layer
])

model2 = tf.keras.Sequential([
    base_model,
    global_average_layer,
    disc_layer1,
    disc_layer2,
    prediction_layer
])

model3 = tf.keras.Sequential([
    base_model,
    global_average_layer,
    disc_layer1,
    disc_layer2,
    prediction_layer
])

model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model2.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model3.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

entrycount = 0
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

for cat in range(clsamount):
    cat_path = os.path.join(DATADIR_AUG, str(cat))
    for img in os.listdir(cat_path):
        entry = cv2.imread(cat_path + '/' + str(img))
        entry = entry / 255.0
        entry = cv2.resize(entry, (80,80), interpolation=cv2.INTER_NEAREST)
        data_list2.append(entry)
        label_list2.append(cat)
        entrycount += 1
        if(entrycount == 1000):
            entrycount = 0
            break
data2 = np.array(data_list2)
labels2 = np.array(label_list2)


x_train, x_split, y_train, y_split = train_test_split(data, labels, test_size = 0.2)
x_test, x_val, y_test, y_val = train_test_split(x_split,y_split, test_size = 0.5)

x_train2, x_split2, y_train2, y_split2 = train_test_split(data2, labels2, test_size = 0.2)
x_test2, x_val2, y_test2, y_val2 = train_test_split(x_split2,y_split2, test_size = 0.5)

history = model.fit(x_train,y_train, batch_size = 32,
                    epochs = 15,
                    shuffle=True,
                    validation_data = (x_val,y_val))

print(f"\n SPLIT1 TESTING \n")

results = model.evaluate(x_test, y_test, batch_size = 32)
model.save_weights('MODEL1/')



history2 = model2.fit(x_train2,y_train2, batch_size = 32,
                    epochs = 15,
                    shuffle=True,
                    validation_data = (x_val2,y_val2))

print(f"\n SPLIT2 TESTING \n")

results2 = model2.evaluate(x_test2,y_test2, batch_size = 16)
model2.save_weights('MODEL2/')

x_train3 = np.concatenate((x_train2, x_val2))
y_train3 = np.concatenate((y_train2, y_val2))


history3 = model3.fit(x_train3,y_train3, batch_size = 32,
                    epochs = 15,
                    shuffle=True,
                    validation_data = (x_val2,y_val2))

print(f"\n SPLIT3 TESTING \n")

results3 = model3.evaluate(x_test2,y_test2, batch_size = 32)
model3.save_weights('MODEL3/')

