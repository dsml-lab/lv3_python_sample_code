# coding: UTF-8

import numpy as np


# 構築したクローン認識器を評価するためのクラス
class LV3_Evaluator:

    def __init__(self, set, extractor):
        # 評価用画像データセット中の全画像から特徴量を抽出して保存しておく
        # 本サンプルコードでは処理時間短縮のため先頭1,000枚のみを用いることにする
        self.samples = []
        for i in range(0, 1000):
            f = set.get_feature(i, extractor)
            self.samples.append((i, f)) # 画像番号と特徴量の組を保存
        self.size = len(self.samples)

    # ターゲット認識器とクローン認識器の出力の一致率（F値）を求める
    #   target: ターゲット認識器
    #   model: クローン認識器
    def calc_accuracy(self, target, model):
        self.target_likelihoods = target.predict_proba(self.samples)
        self.clone_likelihoods = model.predict_proba(self.samples)
        a = self.target_likelihoods >= 0.5
        b = self.clone_likelihoods >= 0.5
        c = np.logical_and(a, b)
        r_avg = 0
        p_avg = 0
        f_avg = 0
        for j in range(0, self.size):
            an = np.sum(a[j])
            bn = np.sum(b[j])
            cn = np.sum(c[j])
            if an != 0:
                r = cn / an
                r_avg += r
            if bn != 0:
                p = cn / bn
                p_avg += p
            if r != 0 or p != 0:
                f = 2 * r * p / (r + p)
                f_avg += f
        r_avg /= self.size
        p_avg /= self.size
        f_avg /= self.size
        return r_avg, p_avg, f_avg
