from keras.preprocessing.image import ImageDataGenerator
import os
import PIL
import scipy

DATADIR = "D:/PyCharm/GTSRB/Train"

datagen = ImageDataGenerator(
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        vertical_flip = True,
        fill_mode = 'nearest')

data_list = []
label_list = []
clsamount = 43
count = 0
for cat in range(clsamount):
    catdir = str(cat)
    for batch in datagen.flow_from_directory(DATADIR+'/'+catdir, batch_size=64, target_size = (80,80), class_mode='sparse', shuffle = True, save_to_dir= DATADIR+'/'+catdir, save_prefix="aug", save_format='png'):
        count += 1
        if (count == 35):
            count = 0
            break
