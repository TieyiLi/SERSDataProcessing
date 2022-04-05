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
    """leave-one-sample-out cross validation splitter"""
    label_set = set(label)
    L = len(label)
    for label_type in label_set:
        test_idx = [i for i in range(L) if label[i] == label_type]
        training_idx = np.setdiff1d(np.arange(L), test_idx)
        yield training_idx, test_idx, label_type

        
def predict_sample_by_counting_maps(pred, map_index_each_sample):
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
    
    return positive_predictions / l

    
def leave_one_sample_out_CV((estimator, X, y, label, group_dict, map_index, num_class='binary', voting_thr=0.5):
    """
    Levae-one-sample-out cross validation.

    Parameters
    ----------
    estimator: classification model
    X: total data matrix
    y: group truth or training label
    label: label of data dictionary
    map_index: map_index of data dictionary
    ori_group: original data_dict['group'] before relabeling
    voting_thr: minimal fraction of diseased required to predict a sample as 
                diseased, default 50%

    Returns
    -------
    None
    """
    r = 0
    true_sample_label = []
    pred_sample_label = []
    scores = []

    for training_idx, test_idx, label_id in leave_one_sample_out(label):
        print('Sample %d val: ' % label_id, end='')
        estimator.fit(X[training_idx], y[training_idx])
        pred = estimator.predict(X[test_idx])
        map_index_each_sample = [map_index[i] for i in test_idx]

        map_preds = predict_sample_by_counting_maps(pred, map_index_each_sample)
        true_of_sample = [key for key in group_dict.keys() if label_id in group_dict[key]][0]
        true_sample_label.append(true_of_sample)

        if num_class == 'binary':
            positive_predictions = np.sum(map_preds==1)
            l = len(map_preds)
            score_of_sample = positive_predictions / l
            sample_pred = threshold_prediction(score_of_sample, voting_thr)
            scores[r] = score_of_sample
            pred_sample_label.append(sample_pred)

            print('%d/%d = %.4f' % (positive_predictions, l, score_of_sample), end='')
            print('----->%d(Truth: %d' % (sample_pred, true_of_sample), end='\n\n')

        elif num_class == 'multi':
            sample_pred = np.argmax(np.bincount(map_preds))
            pred_sample_label.append(sample_pred)
            for c in np.unique(y):
                num = len(map_preds[map_preds == c])
                print('|%d|' % num, end='')
            print('----->%d(Truth: %d)' % (sample_pred, true_of_sample), end='\n\n')

        else:
            raise ValueError('Not supported!')

        r += 1

    print(accuracy_score(true_sample_label, pred_sample_label))
    return [pred_sample_label, true_sample_label, scores]


def threshold_prediction(score, thr):
    if score >= thr:
        return 1
    else:
        return 0
    
    
def KFold_leave_pair_of_samples_out(label, group_dict):
    '''
    This function leaves a pair of negative and positive samples out as
    the validation set while the rest is for training.
    :param label: labels for all spectra given by data_dict['label']
    :param group_dict: group_dict given by data_dict['group_dict']
    :return: yield training and validation sets and the corresponding IDs
    '''
    L = len(label)
    neg_group = group_dict[list(group_dict.keys())[0]]
    pos_group = group_dict[list(group_dict.keys())[1]]
    np.random.shuffle(neg_group)
    np.random.shuffle(pos_group)

    for i in range(len(neg_group)):
        neg_test_idx = np.where(label==neg_group[i])[0]
        pos_test_idx = np.where(label==pos_group[i])[0]
        training_idx = np.setdiff1d(np.setdiff1d(np.arange(L), neg_test_idx), pos_test_idx)
        yield training_idx, [neg_test_idx, pos_test_idx], [neg_group[i], pos_group[i]]


def predict_map_by_counting_spec(pred, map_index):
    '''
    This function gives predictions for every map based on counting the spectra included, threshold of 0.5
    is used by default.
    :param pred: predictions for every spectrum given by classifier
    :param map_index: corresponding map_index for the above predictions
    :return: predictions for the maps belonging to the given sample and total number of maps
    '''
    map_prediction = []
    map_index_set = set(map_index)
    L = len(pred)
    l = len(map_index_set)
    for map in map_index_set:
        idx = [i for i in range(L) if map_index[i] == map]
        map_pred = pred[idx]
        most_common_ele = np.argmax(np.bincount(map_pred))
        map_prediction.append(most_common_ele)
    return np.array(map_prediction), l


def leave_pair_of_samples_out_CV(estimator, X, y, label, group_dict, map_index=None, thr=.5):
    '''
    Performing leave pair of samples out cross validation.
    :param estimator: classifier
    :param X: data matrix
    :param y: ground truth, before relabeling
    :param label: sample labels given by data_dict['label']
    :param group_dict: group_dict given by data_dict['group_dict']
    :param map_index: None for one-spectrum-per-map; data_dict['map_index'] for multiple-spectra-per-map
    :param thr: voting threshold
    :return: predictions for every sample, ground truth for every sample, scores for every sample
    '''
   r = 0
    neg, pos = list(group_dict.keys())[0], list(group_dict.keys())[1]
    sample_number = len(set(label))
    true_sample_label = np.empty(shape=(sample_number, ))
    pred_sample_label = np.empty(shape=(sample_number, ))
    score = np.empty(shape=(sample_number, ))
    sample_label_id = np.empty(shape=(sample_number, ))

    for training_idx, test_idx, test_label_id in KFold_leave_pair_of_samples_out(label, group_dict):

        estimator.fit(X[training_idx], y[training_idx])
        # Spectra predictions for the two validation samples
        neg_pred, pos_pred = estimator.predict(X[test_idx[0]]), estimator.predict(X[test_idx[1]])

        # Determine the sample type based on the counting
        if map_index is None:
            neg_map_num, pos_map_num = len(neg_pred), len(pos_pred)
            neg_pos_num, pos_pos_num = sum(neg_pred==pos), sum(pos_pred==pos)
            score_neg, score_pos = neg_pos_num / neg_map_num, pos_pos_num / pos_map_num
        else:
            neg_map_index = [map_index[s] for s in test_idx[0]]
            pos_map_index = [map_index[t] for t in test_idx[1]]
            neg_map_pred, neg_map_num = predict_maps_by_counting_spec(neg_pred, neg_map_index)
            pos_map_pred, pos_map_num = predict_maps_by_counting_spec(pos_pred, pos_map_index)
            neg_pos_num, pos_pos_num = sum(neg_map_pred==pos), sum(pos_map_pred==pos)
            score_neg, score_pos = neg_pos_num / neg_map_num, pos_pos_num / pos_map_num

        neg_pred_sample, pos_pred_sample = \
            threshold_prediction(score_neg, thr, neg, pos), threshold_prediction(score_pos, thr, neg, pos)

        true_sample_label[r], true_sample_label[r+1] = pos, neg
        pred_sample_label[r], pred_sample_label[r+1] = pos_pred_sample, neg_pred_sample
        score[r], score[r+1] = score_pos, score_neg
        sample_label_id[r], sample_label_id[r+1] = test_label_id[1], test_label_id[0]

        r += 2

        # Inspecting results
        print('Sample %d val: %d/%d=' % (test_label_id[0], neg_pos_num, neg_map_num), end='')
        print('%.5f---->%d(%d)' % (score_neg, neg_pred_sample, neg))
        print('Sample %d val: %d/%d=' % (test_label_id[1], pos_pos_num, pos_map_num), end='')
        print('%.5f---->%d(%d)' % (score_pos, pos_pred_sample, pos), end='\n\n')

    print(accuracy_score(true_sample_label, pred_sample_label))
    return [pred_sample_label, true_sample_label, score]
