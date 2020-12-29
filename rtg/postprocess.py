import numpy as np
import datetime
import os
import torch
import torch.optim as optim
import sys
import matplotlib.pyplot as plt
import argparse

DENSE_FLAG = False
ELAN_FLAG = True
TARGET_TYPE = False


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', type=bool, default=False,
                        help='true: multitask, false: singletask')
parser.add_argument('--gpuid', type=int, default=0)
parser.add_argument('-i', '--input', type=str, default='/mnt/aoni04/jsakuma/data/sota/')
args = parser.parse_args()

# out = 'np/1225/seed2'
out = 'np/1225/mt/seed2'
os.makedirs(out, exist_ok=True)

print('CTC')
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

from utils.utils_ctc import get_dataloader
from models.model import RTG
from utils.trainer import trainer
from utils.eval import quantitative_evaluation, get_F


print('data loading ...')
path = args.input
dataloaders_dict = get_dataloader(path, DENSE_FLAG, ELAN_FLAG, TARGET_TYPE)

if args.task:
    from models.model import MultiTaskRTG
    from utils.trainer_multitask import trainer
    net = MultiTaskRTG(mode=6, input_size=128, input_img_size=65, input_p_size=64, hidden_size=64)
else:
    from models.model import RTG
    from utils.trainer import trainer
    net = RTG(mode=6, input_size=128, input_img_size=65, input_p_size=64, hidden_size=64)



# 学習済み重みの読み込み
PATH = '/mnt/aoni04/katayama/share/work/1224/TGNN/rtg/logs/ctc/1225/seed0/202012250209/epoch_8_loss_0.1708_score0.884.pth'
# PATH = '/mnt/aoni04/katayama/share/work/1224/TGNN/rtg/logs/ctc/1225/seed1/202012250209/epoch_6_loss_0.1762_score0.914.pth'
# PATH = '54/mnt/aoni04/katayama/share/work/1224/TGNN/rtg/logs/ctc/1225/seed2/202012250211/epoch_11_loss_0.1786_score0.897.pth'
PATH = '/mnt/aoni04/katayama/share/work/1224/TGNN/rtg/logs/ctc/1225/mt/seed0/202012261438/epoch_4_loss_0.2507_score0.889.pth'
# PATH = '/mnt/aoni04/katayama/share/work/1224/TGNN/rtg/logs/ctc/1225/mt/seed1/202012261637/epoch_8_loss_0.2381_score0.918.pth'
PATH = '/mnt/aoni04/katayama/share/work/1224/TGNN/rtg/logs/ctc/1225/mt/seed2/202012261637/epoch_5_loss_0.2843_score0.896.pth'

net.load_state_dict(torch.load(PATH, map_location='cpu'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using',device)
net.to(device)
net.eval()

print('Model :', net.__class__.__name__)

optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)


print('train data is ', len(dataloaders_dict['train']))
print('test data is ', len(dataloaders_dict['val']))

# forward 計算
output = trainer(
            net=net,
            mode=6,
            dataloaders_dict=dataloaders_dict,
            optimizer=optimizer,
            scheduler=scheduler,
            only_eval=True
            )

from sklearn.metrics import confusion_matrix, accuracy_score

if args.task:
    y_true_act = output['y_act']
    y_pred_act = output['y_act_pred']
    y_pred_act = np.where(y_pred_act < 0.5, 0, 1)

    u = output['u']
    s = np.asarray([0]+list(u[:-1]-u[1:]))
    # 非発話区間の開始 = -1 , 非発話区間の終了 = 1
    idxs = np.where(s != 0)[0]
    if s[idxs[0]] == 1:
        idxs = idxs[1:]

    y_true = []
    y_pred = []
    # print(y_true_act.max());exit()
    for j in range(0, len(idxs) // 2):
        start, last = idxs[2*j], idxs[2*j + 1]
        if y_pred_act[start:last].max() == y_true_act[start:last].max():
            # correct += 1
            y_true.append(y_true_act[start:last].max())
            y_pred.append(y_pred_act[start:last].max())
        else:
            y_true.append(y_true_act[start:last].max())
            y_pred.append(y_pred_act[start:last].max())
            # no_correct += 1

    fo = open(out+'/report.txt','w')
    print('応答義務推定 acc: {}'.format(accuracy_score(y_true, y_pred)), file=fo)
    print(confusion_matrix(y_true, y_pred), file=fo)
    fo.close()

y_true = output['y']
y_true2 = y_true[1:] - y_true[:-1]
y_true2 = np.maximum(y_true2, 0)

y_prob = output['y_pred']
u_list = output['u']

threshold = 0.5

precision, recall, f1, Distance, LogDistance, (TP, FP, FN), _ = quantitative_evaluation(-1, y_true2, y_prob, u_list, threshold=threshold, 
                                                                    resume=False, output='.', eval_flg=True, only_VAD_ON=True, HO=6)

(R, P, F), (R_log, P_log, F_log) = get_F(Distance, LogDistance, TP, FP, FN)
# 保存
np.save(out+'/rtg-precision.npy',P)
np.save(out+'/rtg-recall.npy', R)
np.save(out+'/rtg-f1.npy', F)

#保存
plt.figure()
hist_ranges = np.arange(-40,40,4)
plt.hist(np.array(Distance) / 50, bins=hist_ranges, histtype='step',linewidth=3.0)
plt.xticks([-30,-20,-10,0,10,20,30],[-1.5,-1.0,0.0,0.5,1.0,1.5])
plt.xlabel('timing err[s]')
plt.savefig(out+'/timing_err.png')