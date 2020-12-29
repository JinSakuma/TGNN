import os
import numpy as np
import math


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
        tp = len(score[score < x])
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
        tp = len(score[score < x])
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

    return (R, P, F), (R_log, P_log, F_log)


def quantitative_evaluation(
                    epoch,
                    y_true,
                    y_pred,
                    u,
                    threshold=0.8,
                    frame=50,
                    resume=True,
                    output='./',
                    eval_flg=False,
                    only_VAD_ON=False,
                    HO=0
                    ):
    target = False
    pred = False
    flag = True
    Distance = []
    logDistance = []
    TP, FP, FN = 0, 0, 0
    start_frame = 0
    for i in range(len(u)-1):
        if u[i] == 0 and u[i+1] == 1:
            start_frame = i  # u が　0->1 になったタイミング

        #  発話中 : 評価対象外
        if u[i] == 0:
            target = False
            pred = False

        #  予測が閾値を超えたタイミング
        if y_pred[i] >= threshold and flag :
            if i > 0 and not only_VAD_ON:
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
            if pred and target:
                TP += 1
                Distance.append((pred_frame-target_frame+HO)*frame)
                logDistance.append(np.log((pred_frame-start_frame+HO)*frame+1)-np.log((target_frame-start_frame)*frame+1))
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

    if len(Distance) > 0:
        score = float(score)/len(Distance)
    else:
        score = -1

    log_score = 0
    for d in logDistance:
        log_score += abs(d**2)

    if len(logDistance) > 0:
        log_score = math.sqrt(float(log_score)/len(logDistance))
    else:
        log_score = -1

    print(TP, FP, FN)
    print('precision:{:.4f}, recall:{:.4f}, f1: {:.4f}, score:{:.4f}, log score:{:.4f}'.format(precision, recall, f1, score, log_score))

    if resume:
        fo = open(os.path.join(output, 'eval_report.txt'), 'a')
        print("""
            Epoch: {}, precision:{:.4f}, recall:{:.4f}, f1: {:.4f}, MAE:{:.4f}, RMSLE:{:.4f}
            """.format(epoch, precision, recall, f1, score, log_score), file=fo)
        fo.close()

    (R, P, F), (R_log, P_log, F_log) = get_F(Distance, logDistance, TP, FP, FN)
    f_log_score = (F_log[14]+F_log[22])/2  # 許容誤差0.7,1.1の時のf1の平均 (F0.7+F1.1)/2

    if eval_flg:
        return precision, recall, f1, Distance, logDistance, (TP, FP, FN), f_log_score

    return precision, recall, f1, Distance, logDistance, f_log_score
