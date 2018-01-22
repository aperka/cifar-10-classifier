#!/usr/bin/env python
#encoding:utf-8
"""
@author:
@time:2017/3/19 11:08
"""
from sklearn.svm import LinearSVC, SVC
from sklearn.externals import joblib
import numpy as np
import glob
import os
import time
from config import *

classifier_cases = {
    1 : {"type": 'linear',
         "path": os.path.join('data', 'models', 'case1')}
}

def classify(train_features_file, test_features_file, classifier_case_path):
    train_fds = []
    train_labels = []
    for img in joblib.load(train_features_file):
        train_fds.append(img[:-1])
        train_labels.append(img[-1])

    test_fds = []
    test_labels = []
    for img in joblib.load(test_features_file):
        test_fds.append(img[:-1])
        test_labels.append(img[-1])

    # If feature directories don't exist, create them
    if not os.path.exists(classifier_case_path):
        if not os.path.exists(os.path.split(classifier_case_path)[0]):
            os.makedirs(os.path.split(classifier_case_path)[0])
        clf = LinearSVC()
        print("Training a Linear SVM Classifier.")
        clf.fit(train_fds, train_labels)
        joblib.dump(clf, classifier_case_path)
    else:
        clf = joblib.load(classifier_case_path)




if __name__ == "__main__":
    from threading import Thread

    for classifier_case in [1]:
        for hog_case in [1,2,3,4,5]:
            train_path = os.path.join("data","{}train.features".format(hog_case))
            test_path = os.path.join("data","{}test.features".format(hog_case))
            class_path = classifier_cases[classifier_case]['path'] + "_{}".format(hog_case)
            Thread(target=classify, kwargs=dict(train_features_file=train_path,
                                                test_features_file=test_path,
                                                classifier_case_path=class_path
                                                )).start()


'''

    t0 = time.time()
    clf_type = 'LIN_SVM'
    fds = []
    labels = []
    num = 0
    total = 0
    for feat_path in glob.glob(os.path.join(train_feat_path, '*.feat')):
        data = joblib.load(feat_path)
        fds.append(data[:-1])
        labels.append(data[-1])

    if clf_type is 'LIN_SVM':
        clf = LinearSVC()
        print("Training a Linear SVM Classifier.")
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        if not os.path.exists(model_path):
            if not os.path.exists(os.path.split(model_path)[0]):
                os.makedirs(os.path.split(model_path)[0])
            joblib.dump(clf, model_path)
        else:
            clf = joblib.load(model_path)

        print("Classifier saved to {}".format(model_path))
        for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
            total += 1
            data_test = joblib.load(feat_path)
            data_test_feat = data_test[:-1].reshape((1, -1))
            result = clf.predict(data_test_feat)
            if int(result) == int(data_test[-1]):
                num += 1
        rate = float(num)/total
        t1 = time.time()
        print('The classification accuracy is %f'%rate)
        print('The cast of time is :%f'%(t1-t0))
'''