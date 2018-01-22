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
         "path": os.path.join('data', 'models', 'case1')},
    2 : {"type": 'poly',
         "path": os.path.join('data', 'models', 'case2'),
         "degree": 3},
    3: {"type": 'poly',
        "path": os.path.join('data', 'models', 'case3'),
        "degree": 2},
    4: {"type": 'rbf',
        "path": os.path.join('data', 'models', 'case4'),
        "????": '????'},
}

def classify(train_features_file, test_features_file, classifier_case_path, classifier_case):
    train_fds = []
    train_labels = []
    for img in joblib.load(train_features_file):
        train_fds.append(img[:-1])
        train_labels.append(img[-1])
    print(len(img))


    # If feature directories don't exist, create them
    if not os.path.exists(classifier_case_path):
        if not os.path.exists(os.path.split(classifier_case_path)[0]):
            os.makedirs(os.path.split(classifier_case_path)[0])

        if classifier_case['type'] == 'linear':
            clf = LinearSVC(max_iter=1000)

        if classifier_case['type'] == 'poly':
            clf = SVC(kernel='poly', degree=classifier_case["degree"], max_iter=10000)

        if classifier_case['type'] == 'rbf':
            clf = SVC(kernel='rbf', max_iter=10000)

        print("Training SVM Classifier.")
        clf.fit(train_fds, train_labels)
        joblib.dump(clf, classifier_case_path)
    else:
        clf = joblib.load(classifier_case_path)

    num = 0
    total = 0

    for img in joblib.load(test_features_file):
        timage_test_feat = img[:-1]
        image_test_label = img[-1]

        result = clf.predict(timage_test_feat.reshape(1, -1))
        total += 1
        if int(result) == int(image_test_label):
            num += 1
    rate = float(num)/total
    print(classifier_case_path + ' acc: {}%'.format(rate))




if __name__ == "__main__":
    from threading import Thread

    for classifier_case in [4]:#, 2, 3]:
        for hog_case in [1,2,3,4,5]:
            train_path = os.path.join("data","{}train.features".format(hog_case))
            train_path = os.path.join("data","{}train.features".format(hog_case))
            test_path = os.path.join("data","{}test.features".format(hog_case))
            class_path = classifier_cases[classifier_case]['path'] + "_{}".format(hog_case)
            Thread(target=classify, kwargs=dict(train_features_file=train_path,
                                                test_features_file=test_path,
                                                classifier_case_path=class_path,
                                                classifier_case=classifier_cases[classifier_case]
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