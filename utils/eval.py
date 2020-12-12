import os
import numpy as np
import math


def quantitative_evaluation(
                    epoch,
                    y_true,
                    y_pred,
                    u,
                    threshold=0.8,
                    frame=50,
                    resume=True,
                    output='./',
                    eval_flg=False
                    ):
    target = False
    pred = False
    flag = True
    Distance = []
    logDistance = []
    TP, FP, FN = 0, 0, 0

    for i in range(len(u)-1):
        if u[i] == 0 and u[i+1] == 1:
            G = i  # u が　0->1 になったタイミング

        #  発話中 : 評価対象外
        if u[i] == 0:
            target = False
            pred = False

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
            if pred and target:
                TP += 1
                Distance.append((pred_frame-target_frame)*frame)
                """
                pred_frame - G ........ u　0->1になったタイミングから予測がどれだけ離れてるか
                target_frame - G ..... u　0->1になったタイミングから正解がどれだけ離れてるか
                """
                logDistance.append((np.log((pred_frame-G)*frame+1)-np.log((target_frame-G)*frame+1))**2)
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
        log_score += abs(d)

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

    if eval_flg:
        return precision, recall, f1, Distance, logDistance, (TP, FP, FN)

    return precision, recall, f1, Distance, logDistance
