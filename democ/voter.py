import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn.preprocessing import OneHotEncoder


class Voter(metaclass=ABCMeta):
    @abstractmethod
    def sampled_fit(self, sampled_features, sampled_likelihoods):
        pass

    @abstractmethod
    def samplable_predict(self, samplable_features):
        pass

    @abstractmethod
    def get_samplable_likelihoods(self):
        pass


class Lv3Voter(Voter):
    """投票者クラス"""

    def __init__(self, model):
        self.model = model
        self.samplable_likelihoods = None

    # クローン認識器の学習
    #   (features, labels): 訓練データ（特徴量とラベルのペアの集合）
    def sampled_fit(self, sampled_features, sampled_likelihoods):
        self.model.fit(features=sampled_features, likelihoods=sampled_likelihoods)  # 学習

    # 未知の二次元特徴量を認識
    #   features: 認識対象の二次元特徴量の集合
    def samplable_predict(self, samplable_features):
        likelihoods = self.model.predict_proba(samplable_features)
        self.samplable_likelihoods = likelihoods  # 予測結果を保持

    def get_samplable_likelihoods(self):
        labels = np.int32(self.samplable_likelihoods >= 0.5)  # 尤度0.5以上のラベルのみがターゲット認識器の認識結果であると解釈する
        return labels
