from math import log, exp

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
import pandas as pd
import matplotlib.pyplot as plt


def cos_sim_to_prob(sim):
    return (sim + 1) / 2  # linear transformation to 0 and 1


def log_prob_to_prob(log_prob):
    return exp(log_prob)


def prob_to_log_prob(prob):
    return log(prob)


def calculate_auroc(all_disease_probs, gt_diseases):
    '''
    Calculates the AUROC (Area Under the Receiver Operating Characteristic curve) for multiple diseases.

    Parameters:
    all_disease_probs (numpy array): predicted disease labels, a multi-hot vector of shape (N_samples, 14)
    gt_diseases (numpy array): ground truth disease labels, a multi-hot vector of shape (N_samples, 14)

    Returns:
    overall_auroc (float): the overall AUROC score
    per_disease_auroc (numpy array): an array of shape (14,) containing the AUROC score for each disease
    '''

    per_disease_auroc = np.zeros((gt_diseases.shape[1],))  # num of diseases
    for i in range(gt_diseases.shape[1]):
        # Compute the AUROC score for each disease
        per_disease_auroc[i] = roc_auc_score(gt_diseases[:, i], all_disease_probs[:, i])

    # Compute the overall AUROC score
    overall_auroc = roc_auc_score(gt_diseases, all_disease_probs, average='macro')

    return overall_auroc, per_disease_auroc


def calculate_total(labels_df, probs_df):
    # Convert the tensor values to numerical values
    labels = labels_df.applymap(lambda x: float(str(x).split('(')[1].split(')')[0]))
    probs = probs_df.applymap(lambda x: float(str(x).split('(')[1].split(')')[0]))

    # Assuming a binary classification problem, get the true labels and predicted probabilities for the positive class
    true_labels = labels[1]
    pred_probs = probs[1]

    # Binarize the predictions with a threshold of 0.5
    pred_labels = (pred_probs >= 0.5).astype(float)

    # Calculate the evaluation metrics
    accuracy = accuracy_score(true_labels, pred_labels,)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    roc_auc = roc_auc_score(true_labels, pred_probs)
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    # Print the evaluation metrics
    print(f"Accuracy: {accuracy:.5f}")
    print(f"Precision: {precision:.5f}")
    print(f"Recall: {recall:.5f}")
    print(f"F1-score: {f1:.5f}")
    print(f"ROC AUC: {roc_auc:.5f}")
    print(conf_matrix)

    # # Calculate the ROC curve
    # fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
    #
    # # Plot the ROC curve
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {roc_auc:.5f})')
    # plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend(loc='lower right')
    # plt.savefig('/home/image023/data/Xplainer-master/roc_curve.png')