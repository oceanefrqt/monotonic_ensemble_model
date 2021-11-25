import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def equiv_key_case(i):
    equiv = {1 : (False, True), 2: (True, True), 3: (False, False), 4: (True, False)}
    return equiv[i]




def equiv_case_key(rev, up):
    equiv = {(False, True):1, (True, True):2, (False, False): 3, (True, False):4 }
    return equiv[(rev, up)]


#PREDICTION FROM A METAMODEL
def pred_metamodel(preds):
    #preds is an array with all the predicted labels from the classifiers belongig to the meta model
    a = preds.count(1)
    b = preds.count(0)
    c = preds.count(-1)

    if c == len(preds):
        print('case of all preds are uncertain. what to do?')
        return None

    if a < b:
        pred = 0
    else:
        pred = 1
    return pred

# Prediction probabilities
def proba_metamodel(preds):
    a = preds.count(1)
    b = preds.count(0)
    if a+b != 0:
        return a/(a+b)
    else:
        print('calcul proba a={}, b={}'.format(a,b))
        return -1


def get_index_positions(list_of_elems, element):
    index_pos_list = []
    index_pos = 0
    while True:
        try:
            index_pos = list_of_elems.index(element, index_pos)
            # Add the index position in list
            index_pos_list.append(index_pos)
            index_pos += 1
        except ValueError as e:
            break
    return index_pos_list

def unclassified_points(labels, probas):
    print('probas {}'.format(probas))
    un_pts = get_index_positions(probas, -1)
    print('unclassified_points {}'.format(un_pts))
    new_labels = deepcopy(labels)
    new_probas = deepcopy(probas)
    for i in range(len(probas)):
        if i not in un_pts:
            new_labels.append(labels[i])
            new_probas.append(probas[i])
    return new_labels, new_probas, un_pts





####AUC score
def perf_metrics(y_actual, y_hat,threshold):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(y_hat)):
        if(y_hat[i] >= threshold):
            if(y_actual[i] == 1):
                tp += 1
            else:
                fp += 1
        elif(y_hat[i] < threshold):
            if(y_actual[i] == 0):
                tn += 1
            else:
                fn += 1
    tpr = tp/(tp+fn)
    fpr = fp/(tn+fp)

    return [fpr,tpr]




def auc_score(probas, labels, auc_file=None):
    thresholds = np.arange(0.0, 1.01, 0.01)

    roc_points = []
    for threshold in thresholds:
        rates = perf_metrics(labels, probas, threshold)
        roc_points.append(rates)

    fpr_array = []
    tpr_array = []
    for i in range(len(roc_points)-1):
        point1 = roc_points[i];
        point2 = roc_points[i+1]
        tpr_array.append([point1[0], point2[0]])
        fpr_array.append([point1[1], point2[1]])

    auc3 = sum(np.trapz(tpr_array,fpr_array))+1
    print('Area under curve={}'.format(auc3))

    CI = confidence_interval(auc3, labels)

    if auc_file is not None:
        plt.plot(tpr_array, fpr_array, linestyle='--', color='darkorange', lw = 2, label='ROC curve', clip_on=False)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('AUC={}, CI=[{};{}] '.format(round(auc3,3), CI[0], CI[1]))
        plt.legend(loc="lower right")
        plt.savefig(auc_file)
    return auc3


def confidence_interval(auc, labels):
    N1 = sum(labels==1)
    N2 = sum(labels!=1)
    Q1 = auc/(2-auc)
    Q2 = (2*(auc**2))/(1+auc)
    SE = sqrt((auc*(1-auc)+(N1-1)*(Q1-auc**2)+(N2-1)*(Q2-auc**2))/(N1*N2))
    low_b = auc - 1.96*SE
    high_b = auc + 1.96*SE
    if low_b < 0:
        low_b = 0
    if high_b > 1:
        high_b = 1
    return [round(low_b, 3), round(high_b,3)]
