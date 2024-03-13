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

PATH = "/home/arjun/Downloads/Skin/"
batch_size = 8
H = 512
W = 512

epsilon = 1e-5
smooth = 1

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    #alpha = 0.7
    R1= (true_pos)/(true_pos + false_neg)
    P1= (true_pos)/(true_pos + false_pos)
    theta1 = 1-R1
    theta2 = 1-P1
    #beta = 0.3
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
     
def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)  ## (H, W, 3)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return ori_x, x                                ## (1, 512, 512, 3)

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (H, W)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.int32)                    ## (512, 512)
    return ori_x, x

def save_results(ori_x, ori_y, y_pred, save_image_path):
    line = np.ones((H, 10, 3)) * 255

    ori_y = np.expand_dims(ori_y, axis=-1)  ## (512, 512, 1)
    ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1) ## (512, 512, 3)

    y_pred = np.expand_dims(y_pred, axis=-1)  ## (512, 512, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) ## (512, 512, 3)

    cat_images = np.concatenate([ori_x, line, ori_y, line, y_pred*255], axis=1)
    cv2.imwrite(save_image_path, cat_images)
    
if __name__ == "__main__":
    # Dataset
    IMAGE_SIZE = 512
    test_x = sorted(glob(os.path.join(PATH, "/home/arjun/Downloads/Skin/Image/*")))
    test_y = sorted(glob(os.path.join(PATH, "/home/arjun/Downloads/Skin/Mask/*")))

    test_steps = (len(test_x)//batch_size)
    if len(test_x) % batch_size != 0:
        test_steps += 1

    with CustomObjectScope({'bce_ft_loss':bce_ft_loss, 'focal_tversky':focal_tversky,'iou': iou, 'dice_coef': dice_coef, 'jacard_coef':jacard_coef}):
        model = tf.keras.models.load_model("/home/arjun/Downloads/UNet_skin_MHL.h5")
        #
    
    """ Predicting the mask and calculating metrics values. """
    SCORE = []
    TN = []
    FP = []
    FN = []
    TP = [] 

    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Exctracting the image name """
        name = x.split("/")[-1]

        """ Read the image and mask """
        ori_x, x = read_image(x)
        ori_y, y = read_mask(y)

        """ Predicting the mask """
        y_pred = model.predict(x)[0] > 0.5
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred.astype(np.int32)

        """ Saving the predicted mask """
        save_image_path = f"/home/arjun/Downloads/B/{name}"
        save_results(ori_x, ori_y, y_pred, save_image_path)

        """ Flatten the array """
        y = y.flatten()
        y_pred = y_pred.flatten()

        """ Calculating metrics values """
        f1_value = f1_score(y, y_pred, labels=[0, 1], average="binary")
        recall_value = recall_score(y, y_pred, labels=[0, 1], average="binary")
        precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary")
        SCORE.append([name, precision_value, recall_value, f1_value])

    df = pd.DataFrame(SCORE, columns = ["Image Name", "Precision", "Recall", "F1"])
    df.to_csv("/home/arjun/Downloads/B/M_score.csv")
