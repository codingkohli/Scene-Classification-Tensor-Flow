#loading/importing the dependencies
import hyperparameters as hp
import os
from glob import glob
import numpy as np
import cv2
from vgg_model import VGGModel

#importing the tensorflow/tensorpack dependencies
from tensorpack import *
from tensorpack.tfutils.sessinit import get_model_loader
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.dataflow.base import RNGDataFlow

#this loads all the images in a numpy array
def getImgList(dir,name):
    #defining an array to load the images
    imgData = []
    imgList = [(fname, os.path.basename(os.path.dirname(fname))) for fname in glob('%s/%s/*/*' % (dir, name))]
    return imgList


def createLabelDict(imgList) :
    if not imgList :
        return
    label_lookup = dict()
    for label in sorted(set(i[1] for i in imgList)):
        label_lookup[label] = len(label_lookup)
    return label_lookup

# Load images into numpy array
def loadImages(imgList,imgSize,labelLookup,name) :
    idxs = np.arange(len(imgList))
    imgsData = np.zeros((imgSize, imgSize, 3, len(imgList)), dtype=np.float)
    for k in idxs:
        fname, label = imgList[k]
        full_dir = os.path.join(hp.datadir, name)
        fname = os.path.join(full_dir, fname)
        temp = cv2.imread(fname)
        img = cv2.resize(cv2.imread(fname), (imgSize, imgSize))
        img = img / 255.0  # You might want to remove this line for your standardization.
        imgsData[:, :, :, k] = img
    return imgsData


#function to set the configurations of the network
def setParams(trainDataSet,testDataSet):
    model = VGGModel()
    config = TrainConfig(
        model,
        dataflow=trainDataSet,
        callbacks=[ModelSaver(),InferenceRunner(trainDataSet,[ScalarStats('cost'),ClassificationError()])],
        max_epoch = hp.num_epochs)
    return config

# getting the images in a list
trainImgList = getImgList(hp.datadir,'train')
testImgList = getImgList(hp.datadir,'test')

# creating a label list from the train dataset
labelLookup = createLabelDict(trainImgList)

#getting the train and test dataset
trainDataSet = loadImages(trainImgList,hp.img_size,labelLookup,'train')
testDataSet = loadImages(trainImgList,hp.img_size,labelLookup,'test')

#confirming the length of the images loaded
#print(len(trainDataSet))

# getting the model by calling the parameters
config = setParams(trainDataSet,testDataSet)
if not config :
    print(False)
