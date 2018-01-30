
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from urllib.request import urlretrieve
from os.path import isfile, isdir
import tarfile
import os, struct
import matplotlib.pyplot as plt
from IPython.display import display
from tqdm import tqdm
import gzip
import sys
import h5py
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


# In[2]:

def loadMNIST(image_src, label_src, cimg):
    
    """Download MNIST data and return train and label datasets"""
    
    print ('Downloading and unpacking' + image_src)
    gzfname, h = urlretrieve(image_src)
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            # Read magic number.
            if n[0] != 0x3080000:
                raise Exception('Invalid file: unexpected magic number.')
            # Read number of entries.
            n = struct.unpack('>I', gz.read(4))[0]
            if n != cimg:
                raise Exception('Invalid file: expected {0} entries.'.format(cimg))
            crow = struct.unpack('>I', gz.read(4))[0]
            ccol = struct.unpack('>I', gz.read(4))[0]
            if crow != 28 or ccol != 28:
                raise Exception('Invalid file: expected 28 rows/cols per image.')
            # Read data.
            res_image = np.fromstring(gz.read(cimg * crow * ccol), dtype = np.uint8)
    finally:
        os.remove(gzfname)
    print ('Image set done.')
    print ('Downloading and unpacking' + label_src)
    gzfname, h = urlretrieve(label_src)
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            # Read magic number.
            if n[0] != 0x1080000:
                raise Exception('Invalid file: unexpected magic number.')
            # Read number of entries.
            n = struct.unpack('>I', gz.read(4))
            if n[0] != cimg:
                raise Exception('Invalid file: expected {0} rows.'.format(cimg))
            # Read labels.
            res_label = np.fromstring(gz.read(cimg), dtype = np.uint8)
    finally:
        os.remove(gzfname)
    print ('Label set done.')
    
    return (res_image.reshape((cimg, crow, ccol)), res_label.reshape((cimg, 1)))


# In[3]:

def maybe_download(src, filename, force=False):
    
    """Download a file if not present, and make sure it's the right size."""
    
    if force or not os.path.exists(filename):
        print('Attempting to download:', filename) 
        filename, _ = urlretrieve(url + filename, filename)
        print('Download Complete!')
    statinfo = os.stat(filename)
        
    return filename
    
def maybe_untar(filename, force=False):
    
    """Untar the downloaded .tar.gz file"""
    
    root = os.path.splitext(os.path.splitext(filename)[0])[0] 
    print('Extracting data for %s.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
    data_folders = root
    print(data_folders)
    
    return data_folders


# In[23]:

def randomly_concat_img(images, labels, concat_length, image_size):
    
    """Randomly concatenate images and labels of initial size of 28x28 (i.e. MNIST) and paint them to a new canvas
       and resize them"""
    
    assert image_size[0] == image_size[1] , 'Final image dimension must be a square.'
    
    import random as rd
    from PIL import Image
    index = 0
    new_image_resized = np.zeros(np.array(image_size))
    new_label = np.ones([concat_length])*10
    i = 0
    while i <= len(images) - 1:
        # Create randomly numbers of concatenated images
        concat_no = rd.randint(0, concat_length-1) if i < len(images)-concat_length else 0
        print(concat_no)
        total_images = concat_no + 1
        concat_image = images[i]
        concat_label = np.ones([concat_length])*10
        concat_label[0] = labels[i]
        if concat_no > 0:
            for j in np.arange(1, total_images):
                concat_image = np.concatenate((concat_image, images[i+j]), axis=1)
                concat_label[j] = labels[i+j]
            concat_label = np.insert(concat_label, 0, total_images)
            new_label = np.vstack((new_label, concat_label))
            i += concat_no
        else:
            i += 1
            concat_label = np.insert(concat_label, 0, total_images)
            new_label = np.vstack((new_label, concat_label))
        index += 1
            
        # Incorporate the new image to a larger ramdomly sized background (initial picture dimension is 28x28)
        #new_image = np.zeros((28*(rd.randint(1,4)), 28*(total_images + rd.randint(0,5))))
        #insert_position = [rd.randint(0, new_image.shape[0] - 28), rd.randint(0, new_image.shape[1] - 28*(total_images))]
        #new_image[insert_position[0]:insert_position[0] + concat_image.shape[0],\
        #           insert_position[1]:insert_position[1] + concat_image.shape[1]] = concat_image
        
        # Resize the entire picture to image_size
        Im = Image.fromarray(concat_image)
        Im = Im.resize(image_size)
        new_image_resized = np.concatenate((new_image_resized,Im), axis=0) 
        index += 1
    return (new_image_resized.reshape(int(new_image_resized.shape[0]/image_size[0]), image_size[0], image_size[1]),             new_label)


# In[24]:

class DigitStructFile:
    def __init__(self, inf):
        self.inf = h5py.File(inf, 'r')
        self.digitStructName = self.inf['digitStruct']['name']
        self.digitStructBbox = self.inf['digitStruct']['bbox']

    def getName(self,n):
        return ''.join([chr(c[0]) for c in self.inf[self.digitStructName[n][0]].value])

    def bboxHelper(self,attr):
        if (len(attr) > 1):
            attr = [self.inf[attr.value[j].item()].value[0][0] for j in range(len(attr))]
        else:
            attr = [attr.value[0][0]]
        return attr

    def getBbox(self,n):
        bbox = {}
        bb = self.digitStructBbox[n].item()
        bbox['height'] = self.bboxHelper(self.inf[bb]["height"])
        bbox['label'] = self.bboxHelper(self.inf[bb]["label"])
        bbox['left'] = self.bboxHelper(self.inf[bb]["left"])
        bbox['top'] = self.bboxHelper(self.inf[bb]["top"])
        bbox['width'] = self.bboxHelper(self.inf[bb]["width"])
        return bbox
    
    def getDigitStructure(self,n):
        s = self.getBbox(n)
        s['name']=self.getName(n)
        return s

    def getAllDigitStructure(self):
        return [self.getDigitStructure(i) for i in range(len(self.digitStructName))]

    def getAllDigitStructure_ByDigit(self):
        pictDat = self.getAllDigitStructure()
        result = []
        structCnt = 1
        for i in range(len(pictDat)):
            item = { 'filename' : pictDat[i]["name"] }
            figures = []
            for j in range(len(pictDat[i]['height'])):
                figure = {}
                figure['height'] = pictDat[i]['height'][j]
                figure['label']  = pictDat[i]['label'][j]
                figure['left']   = pictDat[i]['left'][j]
                figure['top']    = pictDat[i]['top'][j]
                figure['width']  = pictDat[i]['width'][j]
                figures.append(figure)
            structCnt = structCnt + 1
            item['boxes'] = figures
            result.append(item)
        return result
    
print("Successfully created dictionary of bounding boxes!")


# In[40]:

def generate_dataset(data, folder):

    dataset = np.ndarray([len(data),32,32,1], dtype='float32')
    labels = np.ones([len(data),6], dtype=int) * 10
    for i in np.arange(len(data)):
        filename = data[i]['filename']
        fullname = os.path.join(folder, filename)
        im = Image.open(fullname)
        boxes = data[i]['boxes']
        num_digit = len(boxes)
        labels[i,0] = num_digit
        top = np.ndarray([num_digit], dtype='float32')
        left = np.ndarray([num_digit], dtype='float32')
        height = np.ndarray([num_digit], dtype='float32')
        width = np.ndarray([num_digit], dtype='float32')
        for j in np.arange(num_digit):
            if j < 5: 
                labels[i,j+1] = boxes[j]['label']
                if boxes[j]['label'] == 10: labels[i,j+1] = 0
            else: print('#',i,'image has more than 5 digits.')
            top[j] = boxes[j]['top']
            left[j] = boxes[j]['left']
            height[j] = boxes[j]['height']
            width[j] = boxes[j]['width']
        
        im_top = np.amin(top)
        im_left = np.amin(left)
        im_height = np.amax(top) + height[np.argmax(top)] - im_top
        im_width = np.amax(left) + width[np.argmax(left)] - im_left
        
        im_top = np.floor(im_top - 0.1 * im_height)
        im_left = np.floor(im_left - 0.1 * im_width)
        im_bottom = np.amin([np.ceil(im_top + 1.2 * im_height), im.size[1]])
        im_right = np.amin([np.ceil(im_left + 1.2 * im_width), im.size[0]])

        im = im.crop((im_left, im_top, im_right, im_bottom)).resize([32,32], Image.ANTIALIAS)
        im = np.dot(np.array(im, dtype='float32'), [[0.2989],[0.5870],[0.1140]])
        mean = np.mean(im, dtype='float32')
        std = np.std(im, dtype='float32', ddof=1)
        if std < 1e-4: std = 1.
        im = (im - mean) / std
        dataset[i,:,:,:] = im[:,:,:]

    return dataset, labels


# In[ ]:

def naive_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), strides=1, activation='relu', input_shape=input_shape)) 
    model.add(Conv2D(32, kernel_size=(3,3), strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.10))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.10))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])
    model.summary()
    return model


# In[ ]:

def DNN_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), strides=1, activation='relu', input_shape=input_shape)) 
    model.add(Conv2D(32, kernel_size=(3,3), strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.10))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])
    model.summary()
    return model


# In[ ]:

def Deep_CNN(input_shape, num_classes):    
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3,3), strides=1, activation='relu', input_shape=input_shape)) 
    model.add(Conv2D(32, kernel_size=(3,3), strides=1, activation='relu'))
    model.add(Conv2D(32, kernel_size=(3,3), strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.10))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])
    model.summary()
    return model


# In[ ]:

def high_dropout_model(input_shape, num_classes):    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), strides=1, activation='relu', input_shape=input_shape)) 
    model.add(Conv2D(32, kernel_size=(3,3), strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.20))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.30))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])
    model.summary()
    return model

def final_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), strides=1, activation='relu', input_shape=input_shape)) 
    model.add(Conv2D(32, kernel_size=(3,3), strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.10))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.30))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])
    model.summary()
    return model

def get_integer(result):
    resultInt = str(np.argmax(result[0][0:5]))
    for i in np.arange(1, 6):
        resultInt += ',' + str(np.argmax(result[0][i*11-5:i*11+6])) 
    return resultInt

