# estimate the pairwise distance matrix from VLAD descriptors
# Jorge Guevara
# jorged@br.ibm.com

#USAGE: python pairwiseDistance.py -d VLADperPDFdescriptors/VLAD_ORB_W10.pickle -o distaceMatrices/matrix_ORB_W10

from VLADlib.VLAD import *
from VLADlib.Descriptors import *
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import preprocessing
import numpy 
import pickle
import argparse
import glob
import cv2
import csv


#parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--path", required = True,
	help = "Path to VLAD descritpors")
ap.add_argument("-o", "--output", required = True,
	help = "Path of the output file")
args = vars(ap.parse_args())


#args
path = args["path"]
output = args["output"]
metric = args["scale"]

with open(path, 'rb') as f:
    VLAD_DS=pickle.load(f)

imageID=VLAD_DS[0]
print(imageID)
V=VLAD_DS[1]
l =len(imageID)
print(l)
print(V.shape)
f,d=V.shape
j=0

distaceMatrix=pairwise_distances(V,V,metric="euclidean")

#output
filename=output+".txt"

numpy.savetxt(filename, distaceMatrix, delimiter = ',')  