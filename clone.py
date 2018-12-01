# coding: UTF-8

import sys
import csv
import numpy as np
import skimage
from PIL import Image
from skimage.feature import local_binary_pattern
from sklearn import neighbors

from democ.lv3_clf import LV3UserDefinedClassifierDivide
from democ.sampling import lv3_user_function_sampling_democracy
from labels import LabelTable
from evaluation import LV3_Evaluator

# 再帰上限　nによって変更
sys.setrecursionlimit(500000)

# ラベルリストのファイルパス
# ダウンロード先に応じて適宜変更してください
LABEL_LIST = "lv3_label_list.csv"

# データセットが存在するディレクトリのパス
# ダウンロード・解凍先に応じて適宜変更してください
DATASET_PATH = "../../lv3_dataset/"

# クローン認識器訓練用画像が存在するディレクトリのパス
TRAIN_IMAGE_DIR = DATASET_PATH + "train/"

# クローン認識器評価用画像が存在するディレクトリのパス
VALID_IMAGE_DIR = DATASET_PATH + "valid/"

# ラベル表： ラベル名とラベルIDを相互に変換するための表
# グローバル変数として定義
LT = LabelTable(LABEL_LIST)


# 画像特徴抽出器に相当するクラス
# このサンプルコードでは Local Binary Patterns を抽出することにする（skimageを使用）
class LV3_FeatureExtractor:

    # 画像 img から抽出量を抽出する
    def extract(self, img):
        lbp = local_binary_pattern(img, 8, 1, method="uniform")
        f, bins = np.histogram(lbp, bins=256, range=(0, 255), density=True)
        return np.asarray(f, dtype=np.float32)


# ターゲット認識器への入力対象となる画像データセットを表すクラス
class LV3_ImageSet:

    # 画像ファイル名のリストを読み込む
    def __init__(self, image_dir):
        self.imgfiles = []
        f = open(image_dir + "image_list.csv", "r")
        for line in f:
            filename = line.rstrip()  # 改行文字を削除
            self.imgfiles.append(image_dir + filename)
        f.close()

    # データセットサイズ（画像の枚数）
    def size(self):
        return len(self.imgfiles)

    # n番目の画像を取得
    #   as_gray: Trueなら1-channel画像，Falseなら3-channels画像として読み込む
    def get_image(self, n, as_gray=False):
        if as_gray == True:
            img = Image.open(self.imgfiles[n]).convert("L")
        else:
            img = Image.open(self.imgfiles[n]).convert("RGB")
        img = img.resize((128, 128), Image.BILINEAR)  # 処理時間短縮のため画像サイズを128x128に縮小
        return np.asarray(img, dtype=np.uint8)

    # n番目の画像の特徴量を取得
    #   extractor: LV3_FeatureExtractorクラスのインスタンス
    def get_feature(self, n, extractor):
        img = self.get_image(n, as_gray=True)
        return extractor.extract(img)


# ターゲット認識器を表現するクラス
# ターゲット認識器は画像ID（整数値）とそのクラスラベル（マルチラベル，尤度つき）のリストで与えられるものとする
class LV3_TargetClassifier:

    # ターゲット認識器をロード
    #   filename: ターゲット認識器を表すリストファイルのパス
    def load(self, filename):
        global LT
        self.labels = []
        self.likelihoods = []
        f = open(filename, "r")
        reader = csv.reader(f)
        for row in reader:
            temp_label = []
            temp_likelihood = []
            for i in range(1, len(row), 2):
                temp_label.append(LT.LNAME2ID(row[i]))
                temp_likelihood.append(float(row[i + 1]))
            self.labels.append(np.asarray(temp_label, dtype=np.int32))
            self.likelihoods.append(np.asarray(temp_likelihood, dtype=np.float32))
        f.close()

    # ターゲット認識器として使用中の画像リストのサイズ
    def size(self):
        return len(self.labels)

    # 単一サンプルに対し，各クラスラベルの尤度を返す
    #   feature: 関数LV3_user_function_sampling()でサンプリングした特徴量の一つ一つ
    def predict_once(self, feature):
        global LT
        n = feature[0]
        likelihood = np.zeros(LT.N_LABELS())
        for i in range(0, self.labels[n].shape[0]):
            likelihood[self.labels[n][i]] = self.likelihoods[n][i]
        return np.float32(likelihood)

    # 複数サンプルに対し，各クラスラベルの尤度を返す
    #   features: 関数LV3_user_function_sampling()でサンプリングした特徴量
    def predict_proba(self, features):
        likelihoods = []
        for i in range(0, len(features)):
            l = self.predict_once(features[i])
            likelihoods.append(l)
        return np.asarray(likelihoods, dtype=np.float32)


# # クローン認識器を表現するクラス
# # このサンプルコードでは各クラスラベルごとに単純な 5-nearest neighbor を行うものとする（sklearnを使用）
# # 下記と同型の fit メソッドと predict_proba メソッドが必要
# class LV3_UserDefinedClassifier:
#
#     def __init__(self):
#         global LT
#         self.clfs = []
#         for i in range(0, LT.N_LABELS()):
#             clf = neighbors.KNeighborsClassifier(n_neighbors=5)
#             self.clfs.append(clf)
#
#     def __mold_features(self, features):
#         temp = []
#         for i in range(0, len(features)):
#             temp.append(features[i][1])
#         return np.asarray(temp, dtype=np.float32)
#
#     # クローン認識器の学習
#     #   (features, likelihoods): 訓練データ（特徴量と尤度ベクトルのペアの集合）
#     def fit(self, features, likelihoods):
#         global LT
#         features = self.__mold_features(features)
#         labels = np.int32(likelihoods >= 0.5) # 尤度0.5以上のラベルのみがターゲット認識器の認識結果であると解釈する
#         for i in range(0, LT.N_LABELS()):
#             l = labels[:,i]
#             self.clfs[i].fit(features, l)
#
#     # 未知の特徴量を認識
#     #   features: 認識対象の特徴量の集合
#     def predict_proba(self, features):
#         global LT
#         features = self.__mold_features(features)
#         likelihoods = np.c_[np.zeros(features.shape[0])]
#         for i in range(0, LT.N_LABELS()):
#             p = self.clfs[i].predict_proba(features)
#             likelihoods = np.hstack([likelihoods, np.c_[p[:,1]]])
#         likelihoods = likelihoods[:, 1:]
#         return np.float32(likelihoods)


# # ターゲット認識器に入力する画像特徴量をサンプリングする関数
# #   set: LV3_ImageSetクラスのインスタンス
# #   extractor: LV3_FeatureExtractorクラスのインスタンス
# #   n_samples: サンプリングする特徴量の数
def LV3_user_function_sampling(set, extractor, n_samples=1):

    # まず，画像データセット中の全画像から特徴量を抽出する
    # 本サンプルコードでは処理時間短縮のため先頭5,000枚のみを対象とする
    # 不要なら行わなくても良い
    all_features = []
    for i in range(0, 5000):
        f = set.get_feature(i, extractor)
        all_features.append((i, f)) # 画像番号と特徴量の組を保存

    # 特徴量の集合からn_samples個をランダムに抽出する
    perm = np.random.permutation(n_samples)
    features = []
    for i in range(0, n_samples):
        features.append(all_features[perm[i]])

    return features


# クローン処理の実行
# 第一引数でターゲット認識器を表す画像ファイルが格納されているディレクトリを指定するものとする
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("usage: clone.py /target/classifier/path")
        exit(0)

    # 訓練用画像データセットをロード
    train_set = LV3_ImageSet(TRAIN_IMAGE_DIR)
    print("\nAn image dataset for training a clone recognizer was loaded.")

    # 特徴量抽出器を用意
    extractor = LV3_FeatureExtractor()

    # ターゲット認識器を用意
    target_dir = sys.argv[1]
    if target_dir[-1] != "/" and target_dir[-1] != "\\":
        target_dir = target_dir + "/"
    target = LV3_TargetClassifier()
    target.load(target_dir + "train.csv")  # ターゲット認識器をロード
    print("\nA target recognizer was loaded from {0} .".format(sys.argv[1]))

    # ターゲット認識器への入力として用いる特徴量を用意
    # このサンプルコードではひとまず2,000サンプルを用意することにする
    n = 20
    features = lv3_user_function_sampling_democracy(data_set=train_set, extractor=extractor, n_samples=n, exe_n=n,
                                                    target_model=target, all_image_num=5000, labels_all=LT.labels)
    print("\n{0} features were sampled.".format(n))

    # ターゲット認識器に用意した入力特徴量を入力し，各々の認識結果（各クラスラベルの尤度を並べたベクトル）を取得
    likelihoods = target.predict_proba(features)
    print("\nThe sampled features were recognized by the target recognizer.")

    # クローン認識器を学習
    model = LV3UserDefinedClassifierDivide(labels_all=LT.labels)
    model.fit(features, likelihoods)
    print("\nA clone recognizer was trained.")

    # 学習したクローン認識器の精度を評価
    valid_set = LV3_ImageSet(VALID_IMAGE_DIR)  # 評価用画像データセットをロード
    evaluator = LV3_Evaluator(valid_set, extractor)
    target.load(target_dir + "valid.csv")
    recall, precision, f_score = evaluator.calc_accuracy(target, model)
    print("\nrecall: {0}".format(recall))
    print("precision: {0}".format(precision))
    print("F-score: {0}".format(f_score))
