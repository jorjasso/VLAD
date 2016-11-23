#write to csv
# Jorge Guevara
# jorged@br.ibm.com

from VLADlib.VLAD import *
from VLADlib.Descriptors import *
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
args = vars(ap.parse_args())


#args
path = args["path"]
output = args["output"]

with open(path, 'rb') as f:
    VLAD_DS=pickle.load(f)

imageID=VLAD_DS[0]
print(imageID)
V=VLAD_DS[1]
l =len(imageID)
print(l)
print(V.shape)

with open(output, 'w') as f:
    #writer=csv.writer(f)
    for i in range(l):
        #writer.writerow([imageID[i],V[i]])
        features = [str(fs) for fs in V[i]]
        f.write("%s,%s\n" % (imageID[i], ",".join(features)))

       
    #for im in imageID:
    #	writer.writerow([im,V])
    #thedatawriter = csv.writer(mycsvfile, dialect='mydialect')
    #for row in arrayofdata:
    #   thedatawriter.writerow(row)