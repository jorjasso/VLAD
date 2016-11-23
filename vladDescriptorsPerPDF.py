# Jorge Guevara
# jorged@br.ibm.com
# compute VLAD descriptors per PDF using a visual dictionary and a set of descriptors
# USAGE :
# python vladDescriptorsPerPDF.py  -d dataset -dV visualDictionaryPath -n descriptor -o output
# example :
# python vladDescriptorsPerPDF.py  -d dataset -dV visualDictionary/visualDictionary16SURF.pickle --descriptor SURF -o VLADdescriptors/VLAD_SURF_W16

from VLADlib.VLAD import *
from VLADlib.Descriptors import *
import argparse
import glob
import cv2


#parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to image dataset")
ap.add_argument("-dV", "--visualDictionaryPath", required = True,
	help = "Path to the visual dictionary")
ap.add_argument("-n", "--descriptor", required = True,
	help = "descriptor = SURF, SIFT or  ORB")
ap.add_argument("-o", "--output", required = True,
	help = "Path to where VLAD descriptors will be stored")
args = vars(ap.parse_args())


#args
path = args["dataset"]
pathVD = args["visualDictionaryPath"]
descriptorName=args["descriptor"]
output=args["output"]



#estimating VLAD descriptors for the whole dataset
print("estimating VLAD descriptors per PDF using "+descriptorName+ " for dataset: /"+path+ " and visual dictionary: /"+pathVD)


with open(pathVD, 'rb') as f:
    visualDictionary=pickle.load(f) 

#computing the VLAD descriptors
dict={"SURF":describeSURF,"SIFT":describeSIFT,"ORB":describeORB}  
V, idImages = getVLADDescriptorsPerPDF(path,dict[descriptorName],visualDictionary)

#output
file=output+".pickle"

with open(file, 'wb') as f:
	pickle.dump([idImages, V,path], f)

print("The VLAD descriptors per PDF are  saved in "+file)

