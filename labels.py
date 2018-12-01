# coding: UTF-8


# ラベル名とラベルIDを対応付けるクラス
class LabelTable:

    # ラベルリストをファイルから読み込む
    def __init__(self, filename):
        self.labels = [] # ラベル名配列
        self.dict = {} # ラベル名からラベルIDへのマップを表現する辞書
        i = 0
        f = open(filename, "r")
        for line in f:
            name = line.rstrip() # 改行文字を削除
            self.labels.append(name)
            self.dict[name] = i
            i += 1
        f.close()

    # クラスラベルIDからラベル名を取得
    def ID2LNAME(self, id):
        return self.labels[id]

    # ラベル名からクラスラベルIDを取得
    def LNAME2ID(self, name):
        return self.dict[name]

    # クラスラベルの種類数
    def N_LABELS(self):
        return len(self.labels)
