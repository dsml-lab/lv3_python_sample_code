import math
import random

import numpy as np

from democ.distance import find_furthest_place
from democ.lv3_clf import LV3UserDefinedClassifierDivide
from democ.voter import Lv3Voter, Voter


class Parliament:
    """議会クラス"""

    @staticmethod
    def create_lv3_voters(labels_all):
        voters = [Lv3Voter(model=LV3UserDefinedClassifierDivide(labels_all=labels_all)),
                  Lv3Voter(model=LV3UserDefinedClassifierDivide(labels_all=labels_all))]
        return voters

    def __init__(self, samplable_features, voter1: Voter, voter2: Voter):
        self.voter1 = voter1
        self.voter2 = voter2
        self.samplable_features = samplable_features

    def get_optimal_solution(self, sampled_features):
        self.predict_to_voters()

        # # すべての投票者の投票結果を集計
        # 識別結果1と2の差分をとる
        label_count_arr = np.absolute(
            self.voter1.get_samplable_likelihoods() - self.voter2.get_samplable_likelihoods())

        # 同じ点の値を合計し、1次元行列に変換
        label_count_arr = label_count_arr.max(axis=1)

        max_value = np.amax(label_count_arr)
        index_list = np.where(label_count_arr == max_value)[0]
        filtered_samplable_features = []
        for index in index_list:
            filtered_samplable_features.append(self.samplable_features[index])

        opt_feature = find_furthest_place(sampled_features=sampled_features,
                                          samplable_features=filtered_samplable_features)

        self.delete_samplable_features_lv3([opt_feature])

        return [opt_feature]

    def delete_samplable_features_lv3(self, delete_features):
        temp_list = []
        # # サンプリング候補から除外
        for i, able_feature in enumerate(self.samplable_features):
            stay_flag = True
            for delete_feature in delete_features:
                if able_feature[0] == delete_feature[0]:
                    stay_flag = False

            if stay_flag:
                temp_list.append(self.samplable_features[i])

        self.samplable_features = temp_list

    def fit_to_voters(self, sampled_features, sampled_likelihoods):
        self.voter1.sampled_fit(sampled_features=sampled_features, sampled_likelihoods=sampled_likelihoods)
        self.voter2.sampled_fit(sampled_features=sampled_features, sampled_likelihoods=sampled_likelihoods)

    def predict_to_voters(self):
        self.voter1.samplable_predict(samplable_features=self.samplable_features)
        self.voter2.samplable_predict(samplable_features=self.samplable_features)

    def get_discrepancy_rate_arr(self):
        # # すべての投票者の投票結果を集計
        # 識別結果1と2の差分をとる
        samplable_likelihoods_diff = np.absolute(
            self.voter1.get_samplable_likelihoods() - self.voter2.get_samplable_likelihoods())

        # 同じ点の値を合計し、1次元行列に変換
        diff_sum_list = samplable_likelihoods_diff.sum(axis=1)

        return diff_sum_list / np.amax(diff_sum_list)
