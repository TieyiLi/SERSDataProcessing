"""
============================
Multiple validations methods 
============================
This file includes specifically designed leave-maps-out cross validation and leave-sample-out
cross validation as well as the follow-up evaluations.
"""


import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix


def accuracy_by_maps(y_true, y_pred, map_index_test):
    """
    This function computed the validation accuracy with each map as a unit therefore the output
    value equals corrected-predicted-maps/total-maps.

    Parameters
    ----------
    y_ture: ground truth
    y_pred: predictions by training model
    map_index_test: map_index of data dictionary

    Returns
    -------
    None
    """
    if len(map_index_test) == len(y_pred) and len(y_true) == len(y_pred):
    
        L = len(y_pred)
        correct_pred_map = 0
        map_index_test_set = list(set(map_index_test))
        l = len(map_index_test_set)
        for map in map_index_test_set:
            idx = [i for i in range(L) if map_index_test[i] == map]
    
            if len(set(y_true[idx])) != 1:
                raise ValueError('True labels in test map are inconsistent!')
    
            map_true_label = y_true[idx[0]]
            map_pred_label = np.argmax(np.bincount(y_pred[idx]))
            if map_true_label == map_pred_label:
                correct_pred_map += 1

        print(str(correct_pred_map) + '/' + str(l))
        return correct_pred_map / l
    
    else:
        raise ValueError('Inconsistent input dimensions!')
        
def regular_CV(estimator, X, y, splitter, output='accuracy'):
    """
    This function performs the regular cross-validation, thus using each spectra as a unit.

    Parameters
    ----------
    estimator: classification model
    X: total data matrix
    y: group truth or training label
    splitter: cross-validation dataset splitter in sklearn.model_selection module, default
              StratifiedShuffleSplit
    output: show validation metrics for each round

    Returns
    -------
    accuracy
    """
    
    round = 1
    for training_idx, val_idx in splitter.split(X, y):
        print('Cross validation round %d: ' % round, end='')
        X_train, y_train = X[training_idx], y[training_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_val)
        if output == 'accuracy':
            acc_val = accuracy_score(y_val, y_pred)
            print(acc_val)
        elif output == 'sensi&speci':
            if len(set(y)) == 2:
                cfm = confusion_matrix(y_val, y_pred)
                sensi = cfm[1,1] / (cfm[1,1] + cfm[1,0])
                speci = cfm[0,0] / (cfm[0,0] + cfm[0,1])
                print([sensi, speci])
            else:
                raise ValueError('Sensitivity and specificity only apply for binary classification!')
        else:
            raise ValueError('Evaluation metrics not defined!')
        round += 1
        

def K_fold_maps_split(map_index, k):
    """
    This function yield training data and valiation data by KFold spiltting with 
    each map as a unit.

    Parameters
    ----------
    map_index: map_index of data dictionary
    k: number of folds to be splitted

    Yield
    -----
    training spectra indexes, validation spectra indexes
    """
    map_index_set = list(set(map_index))
    L = len(map_index)
    tot_idx = np.arange(L)
    splitter = KFold(n_splits=k)
    for _, test_maps_idx in splitter.split(map_index_set):
        test_maps = [map_index_set[j] for j in test_maps_idx]
        # print(test_maps)
        test_idx = [i for i in range(L) if map_index[i] in test_maps]
        training_idx = np.setdiff1d(tot_idx, test_idx)
        yield training_idx, test_idx


def leave_maps_out_CV(estimator, X, y, map_index, splitter=K_fold_maps_split, **kwargs):
    """
    Levae-maps-out cross validation.

    Parameters
    ----------
    estimator: classification model
    X: total data matrix
    y: group truth or training label
    map_index: map_index of data dictionary
    splitter: K_fold_maps_split function defined above

    Returns
    -------
    Total predictions
    """
    
    if len(X) == len(y) and len(y) == len(map_index):
        # tot_maps = len(set(map_index))
        pred = np.ones_like(y) * -1
        i = 1
        for training_idx, test_idx in splitter(map_index, **kwargs):
            print('KFold round %d' % i)
            estimator.fit(X[training_idx], y[training_idx])
            idx_pred = estimator.predict(X[test_idx])
            pred[test_idx] = idx_pred
            map_index_test = [map_index[j] for j in test_idx]

            acc_val = accuracy_by_maps(y[test_idx], idx_pred, map_index_test)
            
            i += 1
        return pred
    else:
        raise ValueError('Inconsistent data shape!')

def leave_one_sample_out(label):
    """leave-one-sample-out cross valiation splitter"""
    label_set = set(label)
    L = len(label)
    for label_type in label_set:
        test_idx = [i for i in range(L) if label[i] == label_type]
        training_idx = np.setdiff1d(np.arange(L), test_idx)
        yield training_idx, test_idx, label_type

def predict_sample_by_counting_maps(pred, map_index_each_sample, voting_thr):
    """
    This function summarizes the predictions of spectra then maps and generates the 
    prediction for each sample. The determinations of maps and samples are both based
    on voting rule.

    Parameters
    ----------
    estimator: classification model
    X: total data matrix
    y: group truth or training label
    map_index: map_index of data dictionary
    splitter: K_fold_maps_split function defined above
    voting_thr: minimal fraction of diseased required to predict a sample as 
                diseased

    Returns
    -------
    Total predictions
    """
    L = len(pred)
    map_counting = []
    map_index_set = set(map_index_each_sample)
    l = len(map_index_set)
    for map in map_index_set:
        idx = [i for i in range(L) if map_index_each_sample[i] == map]
        spectra_pred = pred[idx]
        most_common_ele = np.argmax(np.bincount(spectra_pred))
        map_counting.append(most_common_ele)

    map_counting = np.array(map_counting)
    positive_predictions = np.sum(map_counting == 1)
    print(str(positive_predictions) + '/' + str(l), end='  ')

    if positive_predictions / l >= voting_thr:
        return 1
    else:
        return 0

def leave_one_sample_out_CV(estimator, X, y, label, map_index, ori_group, voting_thr=0.5):
    """
    Levae-one-sample-out cross validation.

    Parameters
    ----------
    estimator: classification model
    X: total data matrix
    y: group truth or training label
    label: label of data dictionary
    map_index: map_index of data dictionary
    voting_thr: minimal fraction of diseased required to predict a sample as 
                diseased, default 50%

    Returns
    -------
    None
    """
    
    step = 0
    sample_number = len(set(label))
    true_sample_label = np.empty(shape=(sample_number, ))
    pred_sample_label = np.empty(shape=(sample_number, ))

    for training_idx, test_idx, label_id in leave_one_sample_out(label):
        print('Sample %d validation: ' % label_id, end='')
        estimator.fit(X[training_idx], y[training_idx])
        pred = estimator.predict(X[test_idx])
        map_index_each_sample = [map_index[i] for i in test_idx]
        
        
        prediction_of_sample = predict_sample_by_counting_maps(pred, map_index_each_sample, voting_thr)
        true_of_sample = original_group[test_idx]
        if len(set(true_of_sample)) != 1:
            raise ValueError('True labels in test map are inconsistent!')
        
        true_sample_label[step] = true_of_sample[0]
        pred_sample_label[step] = prediction_of_sample
        step += 1
        print(str(prediction_of_sample) + '(' + str(true_of_sample[0]) + ')', end='\n\n')

    print(accuracy_score(true_sample_label, pred_sample_label))
