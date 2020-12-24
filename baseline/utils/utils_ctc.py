import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from utils.utils import extract_label, add_active, np_to_dataframe
from utils.utils import add_hang_over, u_t_maxcut, make_pack, make_target, MyDataLoader


def setup(PATH, file_id, dense_flag=False, elan_flag=False):
    feature_dict = {'feature': [], 'voiceA': [], 'voiceB': [], 'img': [], 'phonemeA': [], 'phonemeB': [], 'u': []}
    for i, idx in enumerate(tqdm(file_id)):
        feature_file = os.path.join(PATH, 'feature_50_p', '{}.feature.csv'.format(idx))
        gaze_file = os.path.join(PATH, 'img_middle64', '{}.img_middle64.npy'.format(idx))
        img_middle_feature_fileA = os.path.join(PATH, 'spec', '{}.A.npy'.format(idx))
        img_middle_feature_fileB = os.path.join(PATH, 'spec', '{}.B.npy'.format(idx))
        phoneme_middle_feature_fileA = os.path.join(PATH, 'phoneme_ctc_1112', '{}A.npy'.format(idx))
        phoneme_middle_feature_fileB = os.path.join(PATH, 'phoneme_ctc_1112', '{}B.npy'.format(idx))
        u_file = os.path.join(PATH, 'u_t_50', '{}.feature.csv'.format(idx))

        df = extract_label(feature_file)
        df_u = pd.read_csv(u_file)[['u_pred', 'u_B_pred']]

        if elan_flag:
            df = add_active(df, feature_file)

        gaze = np.load(gaze_file)
        gaze_new = np.zeros([len(gaze)*2, 64])
        for j in range(len(gaze)):
            gaze_new[j*2] = gaze[j]
            gaze_new[j*2+1] = gaze[j]

        gaze = pd.DataFrame(gaze_new)
        img = np_to_dataframe(img_middle_feature_fileA)
        imgB = np_to_dataframe(img_middle_feature_fileB)

        phonA = np.load(phoneme_middle_feature_fileA, allow_pickle=True)
        phonB = np.load(phoneme_middle_feature_fileB, allow_pickle=True)
        min_len = min([len(df), len(df_u), len(img), len(gaze), len(phonA), len(phonB)])
        # vad_file と長さ調整
        feature_dict['feature'].append(df[:min_len])
        feature_dict['u'].append(df_u[:min_len])
        feature_dict['voiceA'].append(img[:min_len])
        feature_dict['voiceB'].append(imgB[:min_len])
        feature_dict['img'].append(gaze[:min_len])
        feature_dict['phonemeA'].append(phonA[:min_len])
        feature_dict['phonemeB'].append(phonB[:min_len])

    return feature_dict


def preprocess(feat, target_type, phase='train'):
    pack = []

    OC, YC = 0, 0
    threshold = 1.0
    for i in range(len(feat['feature'])):
        u = (1 - feat['feature'][i]['utter_A'].values) * (1 - feat['feature'][i]['utter_B'].values)  # 非発話度真値 AもBもOFFなら u=1
        u = add_hang_over(u)
        # system の顔向き情報の入れ方
        # 画像特徴量も使う際に使用
        if not target_type:
            target = feat['feature'][i]['target'].map(lambda x: 0 if x == 'A' else 1).values
            target = target.reshape(len(target), 1)
        else:
            target = make_target(feat['feature'][i])

        img = feat['img'][i].values
        img = np.append(target, img, axis=1)

        # 教師ラベル
        y = feat['feature'][i]['action'].map(lambda x: threshold if x in ['Active','Passive'] else 0)
        action = feat['feature'][i]['action'].map(lambda x: 1 if x == 'Passive' else 0)
        action_c = feat['feature'][i]['action'].map(lambda x: 1 if x in ['Active-Continue','Passive-Continue'] else 0).values
        y[0] = 0.  # start は予測不可
        u[0] = 0
        action[0] = 0
        u[action_c==1] = 1.
        y = np.asarray(y)
        action = np.asarray(action)
        YC += len(y[y > 0])
        over_flg = False
        start_i = 0
        act_tmp = 0
        timing_label = np.ones(len(y)) * 60.0
        u_interval = np.ones(len(y)) * 60.0
        point = 0
        for j in range(len(y)-1):
            if y[j] > 0 and u[j] == 0:
                y[j] = 0
                action[j] = 0
                act_tmp = action[j]
                OC += 1
                start_i = j
                over_flag = True

            if u[j] == 0 and u[j+1] == 1:
                if over_flg:
                    if abs(j-start_i) > 40:
                        over_flag = False
                        action_c[start_i:j+1] = 0
                    else:
                        y[j+2] = threshold
                        action[j+2] = act_tmp
                        act_tmp = 0
                        u[j+2] = 1
                        action_c[start_i:j+1] = 0
                        action[start_i:j+1] = 0
#                         action_c[j+3:j+3+(j-start_i)] = 1
                        over_flg = False
                
                point = j + 1

            if y[j] > 0 and u[j] == 1:
                timing_label[point] = min(j - point, 58)

            if u[j] == 1 and u[j+1] == 0:
                u_interval[point] = min(j + 1 - point, 60)

        from collections import Counter
        # print(Counter(u_interval))
        # exit()
        # if phase == 'train':  # sys発話中は u(t) を下げる
        #     u[action_c == 1] = 0.
        # else:
        #     u[action_c == 1] = 1.
        # 非発話区間全体のアクションラベルを統一する
        flg = False
        start = 0
        end = 0
        act_tmp = 0
        for j in range(len(u)-1):
            if u[j] == 0 and u[j+1] == 1:
                start = j+1
                # print(timing_label[j+1])
                
            if y[j] > 0:
                flg = True
                act_tmp = action[j]
            if u[j] == 1 and u[j+1] == 0:
                end = j+1
                if flg:
                    action[start:end] = act_tmp
                    flg = False

        # パッケージングに必要なインデックス(非発話区間の開始箇所) ###重要!!
        s = np.asarray([0]+list(u[:-1]-u[1:]))
        # 非発話区間の開始 = -1 , 非発話区間の終了 = 1
        idxs = np.where(s != 0)[0]
        if s[idxs[0]] == -1:
            idxs = idxs[1:]
        # 正解タイミング
        target_idxs = np.where(y > 0)[0]
        
        # 発話区間の個数分の空リストを生成
        batch_list = [{'voiceA': [], 'voiceB': [], 'img': [], 'phonemeA': [], 'phonemeB': [], 'u_pred': [], 'u': [], 'y': [], 'action': []} for _ in range(len(idxs)//2)]
        # print(len(batch_list))

        for j in range(0, len(idxs) // 2):
            start, last = idxs[2*j], idxs[2*j + 1]

            if last - start < 4:
                start = max(0, last-20)
            # start = last - 20
            batch_list[j]['voiceA'] = feat['voiceA'][i].values[start:last]
            batch_list[j]['voiceB'] = feat['voiceB'][i].values[start:last]
            batch_list[j]['img'] = img[start:last]
            batch_list[j]['phonemeA'] = feat['phonemeA'][i][start:last]
            batch_list[j]['phonemeB'] = feat['phonemeB'][i][start:last]
            batch_list[j]['y'] = timing_label[last]
            batch_list[j]['u'] = u_interval[last] #非発話区間の長さ

        a = [batch_list[i]['y'] for i in range(len(batch_list))]
        print(Counter(a))

        pack.append(batch_list)

    print('turn taking timing: {}'.format(YC))
    print('overlap: {}'.format(OC))

    return pack


def get_dataloader(path, dense_flag, elan_flag, target_type, batch_size=1):
    file_id = [file.replace('.feature.csv', '') for file in sorted(os.listdir(os.path.join(path, 'feature_50_p')))]
    train_num = 100
    feat_train = setup(path, file_id[:train_num], dense_flag=dense_flag, elan_flag=elan_flag)
    feat_val = setup(path, file_id[train_num:], dense_flag=dense_flag, elan_flag=elan_flag)

    pack_train = preprocess(feat_train, target_type=target_type, phase='train')
    pack_val = preprocess(feat_val, target_type=target_type, phase='val')

    train_loader = MyDataLoader(pack_train, shuffle=True, batch_size=batch_size)
    val_loader = MyDataLoader(pack_val, shuffle=False, batch_size=batch_size)

    dataloaders_dict = {"train": train_loader, "val": val_loader}

    return dataloaders_dict