import math
import numpy as np
import os
import torch
import argparse
import math
from tqdm import tqdm
from models.model import TGNN

######################################################################
# 設定
######################################################################
"""
mode is 0(vad) or 1(img) or 2(phoneme) or 3(vad & img)
     or 4(vad & phoneme) or 5(img & phoneme) or 6 (vad & img & phoneme)
"""
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, default='/mnt/aoni04/jsakuma/data/sota/')
parser.add_argument('-l', '--lang', type=str, default='ctc', help='ctc or julius')
# parser.add_argument('-l', '--lang', type=str, default='julius', help='ctc or julius')
parser.add_argument('-t', '--task', type=bool, default=False,
                    help='true: multitask, false: singletask')
parser.add_argument('-s', '--seed', type=int, default=0)
parser.add_argument('--target_type', action='store_true',
                    help='if True, target shape is 3(A,B,unknown), False is 1(A/B)')
parser.add_argument('-o', '--out', type=str, default='./results/')
parser.add_argument('-e', '--epoch', type=int, default=30)
parser.add_argument('-r', '--resume', type=str, default=True)
parser.add_argument('--hang', type=str, default=False)
parser.add_argument('--gpuid', type=int, default=0)
parser.add_argument('--weight', type=str,
                    default='/mnt/aoni04/katayama/share/SPEC/epoch_20_acc0.887_loss0.266_ut_train.pth')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

# 追加アノテーション
METHOD = 'ctc_elan_false2'
# METHOD = 'multitask_ctc_'
IDX = 0

model_paths = [
              'logs/ctc/0112_not_add/vip0.4/seed0/0/epoch_10_loss_0.1471_score42.950.pth',
              'logs/ctc/0112_not_add/vip0.4/seed1/0/epoch_2_loss_0.1323_score43.737.pth',
              'logs/ctc/0112_not_add/vip0.4/seed2/0/epoch_2_loss_0.1408_score43.995.pth'
              ]


# model_paths = [
#               'logs/multitask/ctc/0107/vip0.4/seed0/0/epoch_25_loss_0.7107_score42.148.pth',
#               'logs/multitask/ctc/0107/vip0.4/seed1/0/epoch_3_loss_0.4659_score43.246.pth',
#               'logs/multitask/ctc/0107/vip0.4/seed2/0/epoch_17_loss_0.7305_score42.954.pth'
#               ]

modes = [6, 6, 6]
# modes = [0, 1, 2, 3, 4, 5, 6]
# modes = [2, 4, 5, 6]
label_list = ['0', '1', '2']
# label_list=['phoneme', 'vad_phoneme', 'img_phoneme', 'vad_img_phoneme']
# label_list=['vad', 'img', 'phoneme', 'vad_img', 'vad_phoneme', 'img_phoneme', 'vad_img_phoneme']
# label_list=['vad', 'img', 'phoneme', 'vad_img', 'vad_phoneme', 'img_phoneme', 'vad_img_phoneme']
ELAN_FLAG = True
DENSE_FLAG = False
TARGET_TYPE = False

out_dir = args.out
os.makedirs(out_dir, exist_ok=True)

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
                    
if args.task:
    from models.model_multitask import TGNN
else:
    from models.model import TGNN

# モデル設定
input_size = 128
input_img_size = 65
hidden_size = 64
if args.lang == 'ctc':
    print('CTC')
    from utils.utils_ctc import get_dataloader
    input_p_size = 64
    ctc_flg = True
else:
    print('Julius')
    from utils.utils_julius import get_dataloader
    input_p_size = 45
    ctc_flg = False

# データの用意
dataloaders_dict = get_dataloader(args.input, DENSE_FLAG, ELAN_FLAG, TARGET_TYPE)


def quantitative_evaluation(
                    y_true,
                    y_pred,
                    u,
                    threshold=0.8,
                    frame=50,
                    resume=True,
                    output='./',
                    eval_flg=False
                    ):

    count = 0
    target = False
    pred = False
    flag = True
    Distance = []
    logDistance = []
    TP, FP, FN = 0, 0, 0
    u_t_count = 0

    for i in range(len(u)-1):
        if u[i] == 0 and u[i+1] == 1:
            start = i  # u が　0->1 になったタイミング
            # print(G)
        #  発話中 : 評価対象外
        if u[i] == 0:
            target = False
            pred = False
            u_t_count = 0

        #  予測が閾値を超えたタイミング
        if y_pred[i] >= threshold and flag:
            if i > 0:
                if u[i] == 0 and y_pred[i-1] < threshold:
                    FP += 1
            if u[i] > 0:
                pred = True
                flag = False
                pred_frame = i

        #  正解ラベルのタイミング
        if y_true[i] > 0:
            target = True
            target_frame = i

        #  u_t が 1→0 に変わるタイミング or u(t)=1 が 一定以上続いた時
        if (u[i] == 1 and u[i+1] == 0):
            flag = True
            count += 1
            u_t_count = 0
            if pred and target:
                TP += 1
                Distance.append((pred_frame-target_frame)*frame)
                logDistance.append(np.log((pred_frame-start)*frame+1)-np.log((target_frame-start)*frame+1))
            elif pred:
                FP += 1
            elif target:
                FN += 1

    if TP > 0:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = precision * recall * 2 / (precision + recall)
    else:
        precision = recall = f1 = 0

    score = 0
    for d in Distance:
        score += abs(d)

    score = float(score)/len(Distance)

    log_score = 0
    for d in logDistance:
        log_score += abs(d**2)

    log_score = math.sqrt(float(log_score)/len(logDistance))

    print(TP, FP, FN)
    print('precision:{:.4f}, recall:{:.4f}, f1: {:.4f}, score:{:.4f}, log score:{:.4f}'.format(precision, recall, f1, score, log_score))

    if resume:
        fo = open(os.path.join(output, 'eval_report.txt'), 'a')
        print("""
            precision:{:.4f}, recall:{:.4f}, f1: {:.4f}, score:{:.4f}, log score:{:.4f}
            """.format(precision, recall, f1, score, log_score), file=fo)
        fo.close()

    if eval_flg:
        return precision, recall, f1, Distance, logDistance, (TP, FP, FN)

    return precision, recall, f1, Distance, logDistance


def get_F(Distance, LogDistance, TP, FP, FN):
    score = []
    for d in Distance:
        score.append(abs(d))

    score = np.asarray(sorted(score))
    X = [i*50 for i in range(61)]
    F = []
    R = []
    P = []
    for x in X:
        tp = len(score[score <= x])
        fn = len(score[score > x])
        precision = tp/(tp+FP)
        recall = tp/(tp+fn+FN)
        if precision+recall == 0:
            f1 = 0
        else:
            f1 = precision * recall * 2 / (precision + recall)

        R.append(recall)
        P.append(precision)
        F.append(f1)

    score = []
    X = [i*0.05 for i in range(61)]
    for d in LogDistance:
        score.append(abs(d))

    score = np.asarray(sorted(score))
    F_log = []
    R_log = []
    P_log = []
    for x in X:
        tp = len(score[score <= x])
        fn = len(score[score > x])
        precision = tp/(tp+FP)
        recall = tp/(tp+fn+FN)
        if precision+recall == 0:
            f1 = 0
        else:
            f1 = precision * recall * 2 / (precision + recall)

        R_log.append(recall)
        P_log.append(precision)
        F_log.append(f1)
        
    print("score")
    print((F_log[14]+F_log[22]) / 2)

    return (R, P, F), (R_log, P_log, F_log)


def eval_model(model_path, mode):

    net = TGNN(mode=mode,
               input_size=input_size,
               input_img_size=input_img_size,
               input_p_size=input_p_size,
               hidden_size=hidden_size,
               weight_path=args.weight,
               ctc=ctc_flg)

    if mode == 0:
        print("音響")
    elif mode == 1:
        print("画像")
    elif mode == 2:
        print("音素")
    elif mode == 3:
        print("音響+画像")
    elif mode == 4:
        print("音響+音素")
    elif mode == 5:
        print("画像+音素")
    else:
        print("音響+画像+音素")

    net.load_state_dict(torch.load(model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using', device)
    net.to(device)
    net.eval()  # モデルを訓練モードに
    epoch_loss = 0.0  # epochの損失和
    train_cnt = 0
    threshold = 0.8
    u_true = np.array([])
    y_true, y_pred = np.array([]), np.array([])

    with torch.no_grad():
        for batch in tqdm(dataloaders_dict['val']):
            out, a = np.zeros(5), np.zeros(5)
            net.reset_state()
            if mode == 2 or mode >= 4:
                    net.reset_phoneme()

            for i in range(len(batch[0])):
                output_dict = net(batch[0][i], out[-1], a[-1], phase='val')

                u_true = np.append(u_true, batch[0][i]['u'])
                y_true = np.append(y_true, batch[0][i]['y'])
                y_pred = np.append(y_pred, output_dict['y'])

                loss = output_dict['loss']
                if loss != 0 and loss != -1:
                    net.back_trancut()
                    loss = loss.item()

    precision, recall, f1, Distance, LogDistance, (TP, FP, FN) = quantitative_evaluation(y_true, y_pred, u_true, threshold=threshold, resume=False, output='.', eval_flg=True)

    (R, P, F), (R_log, P_log, F_log) = get_F(Distance, LogDistance, TP, FP, FN)
    return (R, P, F), (R_log, P_log, F_log)


######################################################################
# 実行
######################################################################
for i, (path, mode) in enumerate(zip(model_paths, modes)):
    (r, p, f), (r_log, p_log, f_log) = eval_model(path, mode)
    label = label_list[i]
    os.makedirs('./results/{}/data/{}/{}'.format(METHOD, IDX, label), exist_ok=True)

    np.save('./results/{}/data/{}/{}/recall.npy'.format(METHOD, IDX, label), r)
    np.save('./results/{}/data/{}/{}/precision.npy'.format(METHOD, IDX, label), p)
    np.save('./results/{}/data/{}/{}/f1.npy'.format(METHOD, IDX, label), f)
    np.save('./results/{}/data/{}/{}/recall_log.npy'.format(METHOD, IDX, label), r_log)
    np.save('./results/{}/data/{}/{}/precision_log.npy'.format(METHOD, IDX, label), p_log)
    np.save('./results/{}/data/{}/{}/f1_log.npy'.format(METHOD, IDX, label), f_log)
