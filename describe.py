# Jorge Guevara
# jorged@br.ibm.com
# compute SIFT, SURF or ORB descriptors from an image dataset
#   
# USAGE :
# python describe.py --dataset dataset --descriptor descriptorName --output output
# or 
# python describe.py -d dataset -n descriptorName -o output
# example :
# python describe.py --dataset dataset --descriptor SURF --output descriptorSURF
# python describe.py --dataset dataset --descriptor SIFT --output descriptorSIFT
# python describe.py --dataset dataset --descriptor ORB --output descriptorORB

from VLADlib.VLAD import *
from VLADlib.Descriptors import *
import argparse
import glob
import cv2

#parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory that contains the images")
ap.add_argument("-n", "--descriptor", required = True,
	help = "descriptor = SURF, SIFT or  ORB")
ap.add_argument("-o", "--output", required = True,
	help = "Path to where the computed descriptors will be stored")
args = vars(ap.parse_args())


#reading arguments
path = args["dataset"]
descriptorName=args["descriptor"]
output=args["output"]

#computing the descriptors
dict={"SURF":describeSURF,"SIFT":describeSIFT,"ORB":describeORB}
descriptors=getDescriptors(path, dict[descriptorName])

#writting the output
file=output+".pickle"

with open(file, 'wb') as f:
	pickle.dump(descriptors, f)
