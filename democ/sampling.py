import numpy as np
from democ.parliament import Parliament


def extract_features_from_images(data_set, extractor, all_image_count):
    # まず，画像データセット中の全画像から特徴量を抽出する
    # 本サンプルコードでは処理時間短縮のため先頭all_image_count枚のみを対象とする
    all_features = []

    for i in range(0, all_image_count):
        f = data_set.get_feature(i, extractor)
        all_features.append((i, f))  # 画像番号と特徴量の組を保存

    return all_features


def lv3_user_function_sampling_democracy(data_set, extractor, n_samples, target_model, exe_n, labels_all, all_image_num):
    if n_samples <= 0:
        raise ValueError

    elif n_samples == 1:
        all_features = extract_features_from_images(data_set=data_set, extractor=extractor,
                                                    all_image_count=all_image_num
                                                    )

        print('n_samples:' + str(n_samples) + ', ' + 'exe_n:' + str(exe_n))

        perm = np.random.permutation(all_image_num)
        # 最初のランダムな配置
        new_features = []
        for i in range(0, n_samples):
            new_features.append(all_features[perm[i]])

        # ターゲットラベルを取得
        target_likelihoods = target_model.predict_proba(new_features)

        if n_samples == exe_n:
            return new_features
        else:
            voters = Parliament.create_lv3_voters(labels_all=labels_all)
            parliament = Parliament(
                samplable_features=all_features,
                voter1=voters[0], voter2=voters[1])

            parliament.delete_samplable_features_lv3(delete_features=new_features)
            return new_features, target_likelihoods, parliament

    elif n_samples > 1:
        old_features, old_target_likelihoods, parliament = lv3_user_function_sampling_democracy(
            n_samples=n_samples - 1,
            target_model=target_model,
            exe_n=exe_n,
            data_set=data_set,
            extractor=extractor,
            labels_all=labels_all,
            all_image_num=all_image_num
        )

        print('n_samples:' + str(n_samples) + ', ' + 'exe_n:' + str(exe_n))

        parliament.fit_to_voters(sampled_features=old_features, sampled_likelihoods=old_target_likelihoods)
        optimal_features = parliament.get_optimal_solution(sampled_features=old_features)
        features = old_features + optimal_features

        new_target_likelihoods = target_model.predict_proba(optimal_features)
        target_likelihoods = np.vstack((old_target_likelihoods, new_target_likelihoods))

        if n_samples == exe_n:
            return features
        else:
            return features, target_likelihoods, parliament