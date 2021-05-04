# USAGE
# python main.py --dataset dataset/texture_dataset/

# import the necessary packages
from submodules.localbinarypatterns import LocalBinaryPatterns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imutils import paths
import argparse
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
args = vars(ap.parse_args())

# grab the image paths in the input dataset directory
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the local binary patterns descriptor along with
# the data and label lists
print("[INFO] extracting features...")
desc = LocalBinaryPatterns(24, 8)
data = [] # store feature vectors
labels = [] # class labels

# loop over the dataset of images
for imagePath in imagePaths:
	# load the image, convert it to grayscale, and quantify it
	# using LBPs
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)

	# extract the label from the image path, then update the
	# label and data lists; split on path separator for brick, marble, sand
	labels.append(imagePath.split(os.path.sep)[-2])
	# update labels list
	data.append(hist)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
print("[INFO] constructing training/testing split...")
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	random_state=22, test_size=0.25)

# construct the set of hyperparameters to tune
# kernel
	# type of kernel = linear, polynomial, regression basis function
	# c value (upper) hard classifier/no mistakes (overfitting)
	# c value (lower) soft classifier/okay mistakes (underfitting)
parameters = [
	{"kernel":
		["linear"],
		"C": [0.0001, 0.001, 0.1, 1, 10, 100, 1000]},
	{"kernel":
		["poly"],
		"degree": [2, 3, 4],
		"C": [0.0001, 0.001, 0.1, 1, 10, 100, 1000]},
	{"kernel":
		["rbf"], #radial basis function
		"gamma": ["auto", "scale"],
		"C": [0.0001, 0.001, 0.1, 1, 10, 100, 1000]}
]

# tune the hyperparameters via a cross-validated grid search
print("[INFO] tuning hyperparameters via grid search")
# -1 means all cores in parallel
grid = GridSearchCV(estimator=SVC(), param_grid=parameters, n_jobs=-1)
start = time.time()
grid.fit(trainX, trainY)
end = time.time()

# show the grid search information
print("[INFO] grid search took {:.2f} seconds".format(
	end - start))
print("[INFO] grid search best score: {:.2f}%".format(
	grid.best_score_ * 100))
print("[INFO] grid search best parameters: {}".format(
	grid.best_params_))

# grab the best model and evaluate it
print("[INFO] evaluating...")
model = grid.best_estimator_
predictions = model.predict(testX)
print(classification_report(testY, predictions))