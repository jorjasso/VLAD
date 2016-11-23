#write to csv
# Jorge Guevara
# jorged@br.ibm.com

#USAGE: python VLADtoPRJ.py -d VLADperPDFdescriptors/VLAD_ORB_W10.pickle -o VLADperPDFdescriptors_prj/VLAD_ORB_W10.prj

from VLADlib.VLAD import *
from VLADlib.Descriptors import *
from sklearn import preprocessing
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
	help = "Path of csv")
ap.add_argument("-s", "--scale", required = True,
    help = "True or False. Scale the features into the [0,1]-interval")
args = vars(ap.parse_args())


#args
path = args["path"]
output = args["output"]
scale = args["scale"]

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

if scale=="True":
    print("scaling")
    min_max_scaler = preprocessing.MinMaxScaler()
    V = min_max_scaler.fit_transform(V)

with open(output, 'w') as f:
    #writer=csv.writer(f)
    for i in range(l+4):
        #writer.writerow([imageID[i],V[i]])
        if i==0:
            f.write("DY\n")
        elif i==1:
            f.write("%s\n" % str(l))
        elif i==2:
            f.write("%s\n" % str(d))
        elif i==3:
            featuresName=[ str(i)  for i in range(d+1)]
            featuresName=";".join(featuresName)
            f.write("%s\n" % featuresName)
        else:
            features = [str(fs) for fs in V[j]]
            name=str(imageID[j])+".txt"
            f.write("%s;%s;%s\n" % (name, ";".join(features),"0.0"))

            j=j+1

       
    #for im in imageID:
    #	writer.writerow([im,V])
    #thedatawriter = csv.writer(mycsvfile, dialect='mydialect')
    #for row in arrayofdata:
    #   thedatawriter.writerow(row)