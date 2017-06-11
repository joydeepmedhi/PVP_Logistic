#!/usr/bin/env python

"""
Train a SVM to categorize 28x28 pixel images into digits (MNIST dataset).
"""
from __future__ import print_function

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score


import matplotlib.pyplot as plt

def main():
    """Orchestrate the retrival of data, training and testing."""
    data = get_data()

    # Get classifier
    clf = LogisticRegression(C =60, verbose = 60, max_iter = 300)


    print("Start fitting. This may take a while")

    # take all of it - make that number lower for experiments
    examples = len(data['train']['X'])
    clf.fit(data['train']['X'][:examples], data['train']['y'][:examples])

    print(clf)

    import cPickle
    # save the classifier
    with open('my_logistic_classifier.pkl', 'wb') as fid:
        cPickle.dump(clf, fid)   

    #clf2 = pickle.loads(s)

    analyze(clf, data)


def analyze(clf, data):
    """
    Analyze how well a classifier performs on data.

    Parameters
    ----------
    clf : classifier object
    data : dict
    """
    # Get confusion matrix
    from sklearn import metrics
    predicted = clf.predict(data['test']['X'])
    print("Confusion matrix:\n%s" %
          metrics.confusion_matrix(data['test']['y'],
                                   predicted))
    print("Accuracy: %0.4f" % metrics.accuracy_score(data['test']['y'],
                                                     predicted))

    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(data['test']['y'],
                                                     predicted)


    print(precision)
    print(recall)
    print(thresholds)



    # Print example
    try_id = 100
    out = clf.predict(data['test']['X'][:try_id])  # clf.predict_proba
    print("out : %s" % out)
    print("Real: %s" % data['test']['y'][:try_id])

    y_score = clf.predict_proba(data['test']['X'])[:,1]
    
    y_test1 = data['test']['y']

    out1 = clf.predict_proba(data['test']['X'][:try_id])[:,1]  # clf.predict_proba

    out2 = [i * 100 for i in out1]
    out2 = ['%0.3f' % i for i in out2]

    print("Probability Scores: %s" % out2)


    roc(y_test1, y_score)


def roc(test, pred):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(test, pred)
        roc_auc[i] = auc(fpr[i], tpr[i])

    print (roc_auc_score(test, pred))
    print (roc_auc[1])
    plt.figure()
    
    plt.plot(fpr[1], tpr[1], lw=2, label='ROC curve (area = %0.4f)' % roc_auc[1])

    plt.plot([0, 1], [0, 1], color='orange', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.show()




def get_data():
 

    df = pd.read_csv('Train_OMIM.csv')

    df1 = pd.read_csv('Test_OMIM.csv')


    x_train = df[df.columns[1:68]]
    x_test = df1[df1.columns[1:68]]
    y_train = df[df.columns[0]]
    y_test = df1[df1.columns[0]]


    x_train = x_train.values
    x_test = x_test.values

    x_train = [np.array(el).flatten() for el in x_train]

    x_test = [np.array(el).flatten() for el in x_test]

    y_train = y_train.values
    y_test = y_test.values



        
    data = {'train': {'X': x_train, 'y': y_train},
            'test': {'X': x_test,  'y': y_test}}
    return data


if __name__ == '__main__':
    main()