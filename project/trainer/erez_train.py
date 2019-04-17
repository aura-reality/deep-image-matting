#!/usr/bin/env python
# coding: utf-8

# In[10]:


import cv2 as cv2
#import matplotlib.pyplot as plt
import numpy as np

import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, ZeroPadding2D, MaxPooling2D, Concatenate,     Reshape, Lambda
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.utils import plot_model
from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop
from trainer.custom_layers.unpooling_layer import Unpooling
from trainer.utils import overall_loss, get_available_cpus, get_available_gpus
from trainer.data_generator import train_gen


# In[13]:


# Encoder
input_tensor = Input(shape=(320, 320, 4))

x = ZeroPadding2D((1, 1))(input_tensor)
x = Conv2D(64, (3, 3), activation='relu', name='conv1_1')(x)
x = ZeroPadding2D((1, 1))(x)
x = Conv2D(64, (3, 3), activation='relu', name='conv1_2')(x)
orig_1 = x
x = MaxPooling2D((2, 2), strides=(2, 2))(x)

x = ZeroPadding2D((1, 1))(x)
x = Conv2D(128, (3, 3), activation='relu', name='conv2_1')(x)
x = ZeroPadding2D((1, 1))(x)
x = Conv2D(128, (3, 3), activation='relu', name='conv2_2')(x)
orig_2 = x
x = MaxPooling2D((2, 2), strides=(2, 2))(x)

x = Conv2D(128, (1, 1), activation='relu', padding='same', name='deconv6', kernel_initializer='he_normal',
           bias_initializer='zeros')(x)
x = BatchNormalization()(x)
x = UpSampling2D(size=(2, 2))(x)
the_shape = K.int_shape(orig_2)
shape = (1, the_shape[1], the_shape[2], the_shape[3])
origReshaped = Reshape(shape)(orig_2)
# print('origReshaped.shape: ' + str(K.int_shape(origReshaped)))
xReshaped = Reshape(shape)(x)
# print('xReshaped.shape: ' + str(K.int_shape(xReshaped)))
together = Concatenate(axis=1)([origReshaped, xReshaped])
# print('together.shape: ' + str(K.int_shape(together)))
x = Unpooling()(together)

x = Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv5', kernel_initializer='he_normal',
           bias_initializer='zeros')(x)
x = BatchNormalization()(x)
x = UpSampling2D(size=(2, 2))(x)
the_shape = K.int_shape(orig_1)
shape = (1, the_shape[1], the_shape[2], the_shape[3])
origReshaped = Reshape(shape)(orig_1)
xReshaped = Reshape(shape)(x)
together = Concatenate(axis=1)([origReshaped, xReshaped])
x = Unpooling()(together)

x = Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv1', kernel_initializer='he_normal',
           bias_initializer='zeros')(x)
x = BatchNormalization()(x)

x = Conv2D(1, (5, 5), activation='sigmoid', padding='same', name='pred', kernel_initializer='he_normal',
           bias_initializer='zeros')(x)

model = Model(inputs=input_tensor, outputs=x)



# In[14]:


model.compile(loss=overall_loss, optimizer='nadam')


# In[39]:


model.fit_generator(train_gen(), epochs=3)


# In[40]:


x =  model.predict(sample_input)


# In[41]:


cv2.imwrite("data/output/helloworld.png", np.squeeze(x*255))


# In[42]:


#plt.imshow(np.squeeze(x * 255))


# In[44]:


#plt.imshow(img)


# In[ ]:


#get test image
image_name = "GT01.png"

image_path = "data/train/input_training_highres/" + image_name 
trimap_path = "data/train/trimap_training_highres/Trimap1/" + image_name
alpha_path = "data/train/mask/" + image_name


img = cv2.imread(image_path)
trimap = cv2.imread(trimap_path,0)
gt_mask = cv2.imread(alpha_path,0)

img = cv2.resize(img,(320,320))
gt_mask = cv2.resize(gt_mask,(320,320))
trimap = cv2.resize(trimap, (320,320), interpolation = cv2.INTER_NEAREST)


sample_input = np.expand_dims(cv2.merge((img, trimap)), axis = 0)
gt_mask = np.expand_dims(np.expand_dims(gt_mask, axis = 0), axis=3)

#plt.imshow(img)

