
# coding: utf-8

# In[21]:


import cv2
import csv
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# In[22]:


lines = []
with open("./data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
                
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                # For linux
                if batch_sample[0][0] == '/':
                    name = './data/IMG/'+batch_sample[0].split('/')[-1]
                else:
                    name = './data/IMG/'+batch_sample[0].split('\\')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                # Flip image
                image_flipped = np.fliplr(center_image)
                measurement_flipped = -center_angle
                images.append(image_flipped)
                angles.append(measurement_flipped)
                
                # For linux (left)
                if batch_sample[1][0] == '/':
                    name = './data/IMG/'+batch_sample[1].split('/')[-1]
                else:
                    name = './data/IMG/'+batch_sample[1].split('\\')[-1]
                left_image = cv2.imread(name)
                left_angle = float(batch_sample[3])
                images.append(left_image)
                angles.append(left_angle+0.1)
                
                # Flip image
                image_flipped = np.fliplr(left_image)
                measurement_flipped = -(left_angle+0.1)
                images.append(image_flipped)
                angles.append(measurement_flipped)
                
                # For linux (right)
                if batch_sample[2][0] == '/':
                    name = './data/IMG/'+batch_sample[2].split('/')[-1]
                else:
                    name = './data/IMG/'+batch_sample[2].split('\\')[-1]
                right_image = cv2.imread(name)
                right_angle = float(batch_sample[3])
                images.append(right_image)
                angles.append(right_angle-0.1)
                
                # Flip image
                image_flipped = np.fliplr(right_image)
                measurement_flipped = -(right_angle-0.1)
                images.append(image_flipped)
                angles.append(measurement_flipped)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


# In[23]:


model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(3,(5,5),activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(24,(5,5),activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(36,(5,5),activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(48,(3,3),activation="relu"))
model.add(MaxPooling2D())
#model.add(Convolution2D(64,(3,3),activation="relu"))
#model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

BATCH_SIZE = 200

train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

model.compile(loss="mse", optimizer="Adam")
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=1)
model.fit_generator(train_generator, steps_per_epoch=             len(train_samples)/BATCH_SIZE+1, validation_data=validation_generator,             validation_steps=len(validation_samples)/BATCH_SIZE+1, epochs=1)
model.save("model.h5")

