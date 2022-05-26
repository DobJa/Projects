import numpy as np
import tensorflow as tf
import os
import cv2
from time import time
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from keras import metrics
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATADIR = "D:/PyCharm/GTSRB/Train"
DATADIR_AUG = "D:/PyCharm/GTSRB/TrainNorm"

data_list = []
label_list = []
data_list2 = []
label_list2 = []
clsamount = 43

# tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

base_model = tf.keras.applications.ResNet50V2(input_shape = (80,80,3),
                                               include_top = False,
                                               weights = "imagenet",)
base_model.trainable = False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
input_layer = tf.keras.layers.Conv2D(filters = 32, kernel_size = (5,5), activation = "relu", input_shape = (80,80,3))
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

# model2 = tf.keras.Sequential([
#     base_model,
#     global_average_layer,
#     disc_layer1,
#     disc_layer2,
#     prediction_layer
# ])
#
# model3 = tf.keras.Sequential([
#     base_model,
#     global_average_layer,
#     disc_layer1,
#     disc_layer2,
#     prediction_layer
# ])

model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy','RootMeanSquaredError', 'CosineSimilarity'])

# model2.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001),
#               loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#               metrics=['accuracy'])
#
# model3.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001),
#               loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#               metrics=['accuracy'])

entrycount = 0
#for cat in range(clsamount):
#    cat_path = os.path.join(DATADIR, str(cat))
#    for img in os.listdir(cat_path):
#        entry = cv2.imread(cat_path + '/' + str(img))
#        entry = entry / 255.0
#        entry = cv2.resize(entry, (80,80), interpolation=cv2.INTER_NEAREST)
#        data_list.append(entry)
#        label_list.append(cat)
#data = np.array(data_list)
#labels = np.array(label_list)

print(f"starting to load stuff \n \n")
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


#x_train, x_split, y_train, y_split = train_test_split(data, labels, test_size = 0.2, random_state=30)
#x_test, x_val, y_test, y_val = train_test_split(x_split,y_split, test_size = 0.5, random_state=30)

x_train2, x_split2, y_train2, y_split2 = train_test_split(data2, labels2, test_size = 0.2, random_state=30)
x_test2, x_val2, y_test2, y_val2 = train_test_split(x_split2,y_split2, test_size = 0.5, random_state=30)

history = model.fit(x_train2,y_train2, batch_size = 32,
                    epochs = 10,
                    shuffle=True,
                    validation_data = (x_val2,y_val2))

print(f"\n Testing \" the worst hyperparameters \" \n")

results = model.evaluate(x_test2, y_test2, batch_size = 32)

# con_mat = tf.math.confusion_matrix(labels=True, predictions=results).numpy()
# con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
#
# con_mat_df = pd.DataFrame(con_mat_norm,
#                      index = range(clsamount),
#                      columns = range(clsamount))
#
# figure = plt.figure(figsize=(8, 8))
# sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
# plt.tight_layout()
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()




print(history.history.keys())


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# plot for mean
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model root_mean_squared_error')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#plot for KLD
plt.plot(history.history['cosine_similarity'])
plt.plot(history.history['val_cosine_similarity'])
plt.title('model cosine_similarity')
plt.ylabel('cosine_similarity')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save_weights('MODEL1/')



#history2 = model2.fit(x_train2,y_train2, batch_size = 32,
#                    epochs = 15,
#                    shuffle=True,
#                    validation_data = (x_val2,y_val2))

#print(f"\n SPLIT2 TESTING \n")

#results2 = model2.evaluate(x_test2,y_test2, batch_size = 16)
#model2.save_weights('MODEL2/')

#x_train3 = np.concatenate((x_train2, x_val2))
#y_train3 = np.concatenate((y_train2, y_val2))


#history3 = model3.fit(x_train3,y_train3, batch_size = 32,
#                    epochs = 15,
#                   shuffle=True,
#                    validation_data = (x_val2,y_val2))

#print(f"\n SPLIT3 TESTING \n")

#results3 = model3.evaluate(x_test2,y_test2, batch_size = 32)
#model3.save_weights('MODEL3/')

