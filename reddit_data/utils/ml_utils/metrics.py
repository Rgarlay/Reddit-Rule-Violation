from reddit_data.logging.logger import logging
from reddit_data.exception.exception import CustomException
import os,sys

import numpy as np

from sklearn.metrics import accuracy_score, f1_score, recall_score


def evaluate_result(train_labels, train_pred_label, test_labels, test_pred_label):
    try:


        train_accuracy_score = accuracy_score(train_labels, train_pred_label)
        train_recall_score = recall_score(train_labels, train_pred_label)
        train_f1_score = f1_score(train_labels,train_pred_label)

        test_accuracy_score = accuracy_score(test_labels, test_pred_label)
        test_recall_score = recall_score(test_labels, test_pred_label)
        test_f1_score = f1_score(test_labels,test_pred_label)

        evaluation_scores = {'TRAIN METRICS':{'Accuracy':train_accuracy_score,
                                           'Recall':train_recall_score,
                                           'f1_score':train_f1_score},
                           'TEST METRICS':{"Accuracy":test_accuracy_score,
                                           "Recall":test_recall_score,
                                           "f1_score":test_f1_score}}
        
        return evaluation_scores
    except Exception as e:
        raise CustomException(e,sys)