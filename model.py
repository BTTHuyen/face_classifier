"""
This is an example of using the k-nearest-neighbors (KNN) algorithm for face recognition.
When should I use this example?
This example is useful when you wish to recognize a large set of known people,
and make a prediction for an unknown person in a feasible computation time.
Algorithm Description:
The knn classifier is first trained on a set of labeled (known) faces and can then predict the person
in an unknown image by finding the k most similar faces (images with closet face-features under eucledian distance)
in its training set, and performing a majority vote (possibly weighted) on their label.
For example, if k=3, and the three closest face images to the given image in the training set are one image of Biden
and two images of Obama, The result would be 'Obama'.
* This implementation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.
Usage:
1. Prepare a set of images of the known people you want to recognize. Organize the images in a single directory
   with a sub-directory for each known person.
2. Then, call the 'train' function with the appropriate parameters. Make sure to pass in the 'model_save_path' if you
   want to save the model to disk so you can re-use the model without having to re-train it.
3. Call 'predict' and pass in your trained model to recognize the people in an unknown image.
NOTE: This example requires scikit-learn to be installed! You can install it with pip:
$ pip3 install scikit-learn
"""

import math
from sklearn import neighbors, metrics
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import cv2
import time
from skimage.feature import hog
import face_recognition
import numpy  as np
from face_recognition.face_recognition_cli import image_files_in_folder
import glob
from sklearn.model_selection import train_test_split

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def read_data(data):
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    X=[]
    Y=[]
    for class_dir in os.listdir(data):


        print(class_dir)
        if not os.path.isdir(os.path.join(data, class_dir)):
            continue
        img_name = glob.glob(os.path.join(data, class_dir)+"/*.jpg")
        for img in img_name:
            #img_np = Image.open(img).convert("RGB")
            #img_np = face_recognition.load_image_file(img)
            #img_encoding = face_recognition.face_encodings(img_np)[0]
            img_np = cv2.imread(img)
            #img_np = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            #print(img_np)
            fd,hog_f = hog(img_np, orientations=8, pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1), visualize=True, multichannel=True)
            #fd = np.matrix(f)
            last = fd.flatten()
            #X.append(img_encoding)
            X.append(last)
            Y.append(class_dir)
    print(X)
    X_train, X_test, y_train, y_test= train_test_split(X, Y, test_size=0.3)

    return X_train, y_train,X_test,y_test
    #return X,Y

def train(X_train,y_train, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X_train, y_train)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
            print("saved")

    return knn_clf


def predict(X_test,y_test, knn_clf=None):
    predict_label = knn_clf.predict(X_test)
    score = metrics.accuracy_score(y_test, predict_label)
    print(predict_label)
    return score

def predict_img(img_path,model):
    with open(model, 'rb') as f:
        knn_clf = pickle.load(f)
    start = time.time()
    img_np = face_recognition.load_image_file(img_path)
    img_encoding = face_recognition.face_encodings(img_np)
    end = time.time()
    print("time for encoding", end - start)
    predict_lbl = knn_clf.predict(img_encoding)
    #print(predict_lbl)

def main():
    X_train, y_train,X_test,y_test = read_data("database")
    #knn_clf = train(X_train,y_train,"knn_hog.pkl",15)
    start = time.time()
    predict_img("s03_01.jpg","knn_hog.pkl")
    end = time.time()
    print(end - start)
    #score = predict(X_test,y_test,knn_clf)
    #print("score: ", score)
    print("score hog",score)
              
main()