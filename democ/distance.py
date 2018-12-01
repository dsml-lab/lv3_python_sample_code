import unittest

import numpy as np
import random


def calc_distance(feature1, feature2):
    feature_diff = feature1 - feature2
    feature_square = feature_diff**2
    return np.sum(feature_square)


def find_furthest_place(sampled_features, samplable_features):
    nearest_arr = get_furthest_rate_arr(sampled_features=sampled_features, samplable_features=samplable_features)

    max_value = np.amax(nearest_arr)

    print("sampling候補数: " + str(len(nearest_arr)))

    index_list = np.where(max_value == nearest_arr)[0]
    random.shuffle(index_list)

    return samplable_features[index_list[0]]


def get_furthest_rate_arr(sampled_features, samplable_features):
    # サンプリング対象点のすべてに関して、各サンプリング済み点との距離を記録するための行列
    distance_arr = np.zeros((len(samplable_features), len(sampled_features)))

    for i, filtered_feature in enumerate(samplable_features):
        for j, sampled_feature in enumerate(sampled_features):
            distance_arr[i][j] = calc_distance(feature1=filtered_feature[1], feature2=sampled_feature[1])

    # サンプリング対象点のすべてに関して、最近傍のサンプリング済み点との距離を記録する行列
    nearest_arr = np.zeros((len(samplable_features)))
    for i, filtered_feature in enumerate(samplable_features):
        nearest_arr[i] = np.min(distance_arr[i])

    return nearest_arr / np.amax(nearest_arr)