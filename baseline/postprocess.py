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

# out = 'np/weight3/seed2'
out = 'np/mt/weight1/seed2'
os.makedirs(out, exist_ok=True)

print('CTC')
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

from utils.utils_ctc import get_dataloader


print('data loading ...')
path = args.input
dataloaders_dict = get_dataloader(path, DENSE_FLAG, ELAN_FLAG, TARGET_TYPE)

if args.task:
    from models.model import MultiTaskmodel
    from utils.trainer_multitask import trainer
    net = MultiTaskmodel(mode=6, input_size=128, input_img_size=65, input_p_size=64, hidden_size=64)
else:
    from models.model import Basemodel
    from utils.trainer import trainer
    net = Basemodel(mode=6, input_size=128, input_img_size=65, input_p_size=64, hidden_size=64)


# 学習済み重みの読み込み
#singletask
# PATH = '/mnt/aoni04/katayama/share/work/1225/TGNN/baseline/logs/ctc/1225/weight1/seed0/202012250206/epoch_45_loss_0.836.pth'
# PATH = '/mnt/aoni04/katayama/share/work/1225/TGNN/baseline/logs/ctc/1225/weight1/seed1/202012250207/epoch_46_loss_0.833.pth'
# PATH = '/mnt/aoni04/katayama/share/work/1225/TGNN/baseline/logs/ctc/1225/weight1/seed2/202012250207/epoch_44_loss_0.855.pth'
# PATH = '/mnt/aoni04/katayama/share/work/1225/TGNN/baseline/logs/ctc/1225/weight2/seed0/202012251101/epoch_27_loss_1.423.pth'
# PATH = '/mnt/aoni04/katayama/share/work/1225/TGNN/baseline/logs/ctc/1225/weight2/seed1/202012251057/epoch_22_loss_1.531.pth'
# PATH = '/mnt/aoni04/katayama/share/work/1225/TGNN/baseline/logs/ctc/1225/weight2/seed2/202012251057/epoch_41_loss_1.499.pth'
# PATH = '/mnt/aoni04/katayama/share/work/1225/TGNN/baseline/logs/ctc/1225/weight3/seed0/202012250351/epoch_35_loss_2.038.pth'
# PATH = '/mnt/aoni04/katayama/share/work/1225/TGNN/baseline/logs/ctc/1225/weight3/seed1/202012250353/epoch_19_loss_2.134.pth'
# PATH = '/mnt/aoni04/katayama/share/work/1225/TGNN/baseline/logs/ctc/1225/weight3/seed2/202012250353/epoch_57_loss_2.130.pth'

#multitask
PATH = '/mnt/aoni04/katayama/share/work/1225/TGNN/baseline/logs/ctc/1225/weight1/mt/seed0/202012261642/epoch_17_loss_0.808.pth'
PATH = '/mnt/aoni04/katayama/share/work/1225/TGNN/baseline/logs/ctc/1225/weight1/mt/seed1/202012261642/epoch_16_loss_0.829.pth'
PATH = '/mnt/aoni04/katayama/share/work/1225/TGNN/baseline/logs/ctc/1225/weight1/mt/seed2/202012261642/epoch_20_loss_0.821.pth'

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

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

if args.task:
    y_true_act = output['y_act']
    y_pred_act = output['y_act_pred']

    y_pred_act = [0 if p < 0.5 else 1 for p in y_pred_act]

    fo = open(out+'/report.txt','w')
    print('応答義務推定 acc: {}'.format(accuracy_score(y_true_act, y_pred_act)), file=fo)
    print(confusion_matrix(y_true_act, y_pred_act), file=fo)
    fo.close()

def msle(y_true, y_pred):
    # 対数誤差を計算
    return (np.log(y_true+1) - np.log(y_pred+1))

timing_error = list()
MSLE = list()
missDifference = list()
threshold = 60
HO = 6 # hang over

y_pred = np.array([])
feature = np.array(dataloaders_dict['val'])

y_true = output['y']

y_prob = output['y_pred']
u_list = output['u']

# 出力の後処理
for i in range(len(y_true)):
    p = y_prob[i]

    if y_true[i] < threshold:
        if y_prob[i] < threshold:# and y_prob[i] < u_list[i]:
            y_pred = np.append(y_pred, 1)
        else:
            y_pred = np.append(y_pred, 0)

    else:
        if y_prob[i]  < u_list[i]:
            y_pred = np.append(y_pred, 1)
        else:
            y_pred = np.append(y_pred, 0)

    if y_pred[i] and y_true[i] < 60:
        timing_error.append((p - y_true[i])*50)

        MSLE.append(msle(p*50, y_true[i]*50))
        print(f' 予測:{p} 真値:{y_true[i]} msle {msle(p*50, y_true[i]*50)}')

    # precision error
    elif y_pred[i]:
        missDifference.append((p+6)*50)

y_true = np.where(y_true < 60, 1, 0)

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
print('MAE is {},RMSLE is {}'.format(np.abs(timing_error).mean(),np.mean(MSLE)**0.5))


# 許容度に対する性能計算
timing_error = np.array(timing_error)
length = len(y_true[y_true==1])
timeout = 60

Recalls = np.array([len(timing_error[np.abs(timing_error)<=i*50]) / length for i in range(timeout+1)])
print(len(missDifference), len(timing_error))
length = len(missDifference) + len(timing_error)
Precisions = np.array([len(timing_error[np.abs(timing_error)<=i*50]) / length for i in range(timeout+1)])
f1 = 2 * Precisions * Recalls / (Precisions + Recalls)

# 保存
np.save(out+'/base-precision.npy',Precisions)
np.save(out+'/base-recall.npy', Recalls)
np.save(out+'/base-f1.npy', f1)

#保存
hist_ranges = np.arange(0,60,4)
plt.hist(y_prob[(y_prob<60) & (y_true==1)],label='pred', bins=hist_ranges,histtype='step',linewidth=3.0)
plt.hist(output['y'][y_true==1],label='true', bins=hist_ranges,histtype='step',linewidth=3.0)
plt.legend()
plt.xticks([0,10,20,30,40,50,60],[0,0.5,1,1.5,2,2.5,3])
plt.xlabel('timing [s]')
plt.savefig(out+'/timing_dist.png')

#保存
plt.figure()
hist_ranges = np.arange(-40,40,4)
plt.hist(timing_error / 50, bins=hist_ranges, histtype='step',linewidth=3.0)
plt.xticks([-30,-20,-10,0,10,20,30],[-1.5,-1.0,0.0,0.5,1.0,1.5])
plt.xlabel('timing err[s]')
plt.savefig(out+'/timing_err.png')