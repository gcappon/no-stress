import numpy as np
import pandas as pd

import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

# new
import tsfel
import sklearn
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


def getFeatures(single_feature_matrix, cfg_file):
    new_single_feat_matrix = pd.DataFrame()
    for row in single_feature_matrix:
        X_sig = pd.DataFrame(np.hstack(row.T), columns=[""])
        X_sig = X_sig.dropna()  # since na are removed, this should be omitted and also the loop is unnecessary
        X = tsfel.time_series_features_extractor(cfg_file, X_sig, window_size=X_sig.shape[0], verbose=
        False)

        new_single_feat_matrix = pd.concat([new_single_feat_matrix, X])
    return new_single_feat_matrix


def wrapperFeatures(train_dataset, test_dataset, feature_type):
    test_new_feat_dataset = pd.DataFrame()
    new_feat_dataset = pd.DataFrame()
    # check the domain type of features to extract
    if feature_type != '':
        cfg_file = tsfel.get_features_by_domain(feature_type)
    else:
        cfg_file = tsfel.get_features_by_domain()

    for feat_idx in range(train_dataset.shape[2]):
        # for each feature (third dim)
        feat_matrix_train = train_dataset[:, :, feat_idx]
        feat_matrix_test = test_dataset[:, :, feat_idx]

        new_single_feat_matrix = getFeatures(feat_matrix_train, cfg_file)

        # Highly correlated features are removed
        corr_features = tsfel.correlated_features(new_single_feat_matrix)
        new_single_feat_matrix.drop(corr_features, axis=1, inplace=True)
        colnames = new_single_feat_matrix.columns
        # print(colnames)

        # Remove low variance features
        ## TODO: (Michele) check this thresholds
        selector = VarianceThreshold()
        new_single_feat_matrix = selector.fit_transform(new_single_feat_matrix)

        # from colnames, keep the ones that are labeled as true (e.g. variance sufficently high) and transform from ndarray to dataframe
        new_single_feat_matrix = pd.DataFrame(data=new_single_feat_matrix,
                                              columns=[colnames[i] for i in range(len(colnames)) if
                                                       selector.get_support()[i]])
        new_colnames = new_single_feat_matrix.columns  # colnames store in order to keep only these one also in the test set

        # rename such that same features names have different "label" related to handcrafted features index
        index_related_column_names = ["handcrafted{}_{}".format(feat_idx, new_single_feat_matrix.columns[idx]) for idx
                                      in range(len(new_single_feat_matrix.columns))]
        new_single_feat_matrix.columns = index_related_column_names

        new_feat_dataset = pd.concat([new_feat_dataset, new_single_feat_matrix], axis=1)

        ### test section ###
        test_new_single_feat_matrix = getFeatures(feat_matrix_test, cfg_file)

        test_new_single_feat_matrix = pd.DataFrame(data=test_new_single_feat_matrix)
        # keep the same features extrcated in training
        test_new_single_feat_matrix = test_new_single_feat_matrix.drop(
            columns=[col for col in test_new_single_feat_matrix if col not in new_colnames])
        # rename such that same features names have different "label" related to handcrafted features index
        index_related_column_names = ["handcrafted{}_{}".format(feat_idx, test_new_single_feat_matrix.columns[idx]) for
                                      idx in range(len(test_new_single_feat_matrix.columns))]
        test_new_single_feat_matrix.columns = index_related_column_names

        test_new_feat_dataset = pd.concat([test_new_feat_dataset, test_new_single_feat_matrix], axis=1)

    return new_feat_dataset, test_new_feat_dataset

names = ['preprocessed_dataset_1.npz', 'preprocessed_dataset_2.npz', 'preprocessed_dataset_3.npz', 'preprocessed_dataset_4.npz', 'preprocessed_dataset_5.npz']

for index, name in enumerate(names):

    data = np.load(os.path.join('data', 'preprocessed', name), allow_pickle=True)

    # training dataset
    x_train = data['x_train'].tolist()
    y_train = data['y_train']
    idx_train = data['idx_train']

    # test dataset
    x_test = data['x_test'].tolist()
    idx_test = data['idx_test']

    data_dict = {}

    for key in x_train.keys():
        train = x_train.get(key)
        # print(ecg)
        print(type(train))
        print(train.shape)

        test = x_test.get(key)

        # train_dataset = handcrafted_features["ECG_features"]
        # test_dataset = dataset_test["hand_crafted_features"]["ECG_features"]
        feature_type = "statistical"

        newTrain, newTest = wrapperFeatures(train, test, feature_type)

        data_dict[key] = [newTrain, newTest]

    # Save
    extraction_name = "feature_extracted_dataset_{}".format(index+1)
    np.savez(os.path.join('data', 'feature_extracted', extraction_name),
             feature_extracted_dataset=data_dict,
             y_train=y_train,
             idx_train=idx_train,
             idx_test=idx_test)

# TODO: (Michele) check of each variable data distribution etc..
# TODO: better code

# example to load the data
# data = np.load(os.path.join('data', 'feature_extracted', 'feature_extracted_dataset_1.npz'), allow_pickle = True)
# datasets = data['feature_extracted_dataset'].tolist()
# ecg_train = datasets.get('ECG_features')[0]
# ecg_test = datasets.get('ECG_features')[0]

