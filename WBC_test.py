from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_FORCE_GPU_GROWTH"] = "true"

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import cv2
from tensorflow.keras.utils import CustomObjectScope
from tqdm import tqdm
import pandas as pd
from glob import glob
from keras.losses import binary_crossentropy
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

PATH = "/home/arjun/Downloads/Blood/"
batch_size = 8

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (512, 512))
    x = x/255.0
    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (512, 512))
    x = np.expand_dims(x, axis=-1)
    return x

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([512, 512, 3])
    y.set_shape([512, 512, 1])
    return x, y

def tf_dataset(x, y, batch=batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset

def read_and_rgb(x):
    x = cv2.imread(x)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    return x

epsilon = 1e-5
smooth = 1

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    R1= (true_pos)/(true_pos + false_neg)
    P1= (true_pos)/(true_pos + false_pos)
    theta1 = 1-R1
    theta2 = 1-P1
    #theta1 = 0.7
    #theta2 = 0.3
    return (true_pos + smooth)/(true_pos + theta1*false_neg + theta2*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)

def bce_ft_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + focal_tversky(y_true, y_pred)
    return loss
   
def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32) 

def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def jacard_coef(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

if __name__ == "__main__":
    # Dataset
    IMAGE_SIZE = 512
    test_x = sorted(glob(os.path.join(PATH, "/home/arjun/Downloads/Blood/Image/*")))
    test_y = sorted(glob(os.path.join(PATH, "/home/arjun/Downloads/Blood/Mask/*")))

    test_dataset = tf_dataset(test_x, test_y, batch=batch_size)

    test_steps = (len(test_x)//batch_size)
    if len(test_x) % batch_size != 0:
        test_steps += 1

    with CustomObjectScope({'bce_ft_loss':bce_ft_loss, 'focal_tversky':focal_tversky, 'iou': iou, 'dice_coef': dice_coef, 'jacard_coef':jacard_coef}):
        model = tf.keras.models.load_model("/home/arjun/Desktop/UNet_WBC_WN_MHL.h5") 
        print(model.summary())      
    
    """ Predicting the mask and calculating metrics values. """
    SCORE = []
    TN = []
    FP = []
    FN = []
    TP = []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Extracing the image name. """
        image_name = x.split("/")[-1]
        
        """ Reading the image and mask. """
        x = read_image(x)
        y = read_mask(y)

        """ Predicting the mask.
            This piece of code also changes if you any other framwork.
        """
        y_pred = model.predict(np.expand_dims(x, axis=0))[0] > 0.5
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred.astype(np.int32)

        """ Flattening the numpy arrays. """
        y = y.flatten()
        y_pred = y_pred.flatten()

        """ Calculating metrics values """
        f1_value = f1_score(y, y_pred, labels=[0, 1], average="binary")
        recall_value = recall_score(y, y_pred, labels=[0, 1], average="binary")
        precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary")
        SCORE.append([image_name, precision_value, recall_value, f1_value])

    """ Saving all the results """
    df = pd.DataFrame(SCORE, columns=["Image", "Precision", "Recall", "F1"])
    df.to_csv("/home/arjun/Downloads/WBC/WN_M_score.csv")
    
    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        x = read_image(x)
        y = read_mask(y)
        y_pred = model.predict(np.expand_dims(x, axis=0))[0] > 0.5
        h, w, _ = x.shape
        white_line = np.ones((h, 10, 3)) * 255.0
        
        all_images = [
            x * 255.0, white_line,
            mask_parse(y)* 255.0, white_line,
            mask_parse(y_pred) * 255.0
        ]  
        image = np.concatenate(all_images, axis=1)
        cv2.imwrite(f"/home/arjun/Downloads/WBC/{i}.png", image)
