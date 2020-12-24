import numpy as np
import datetime
import os
import torch
import torch.optim as optim
import sys
import matplotlib.pyplot as plt

DENSE_FLAG = False
ELAN_FLAG = True
TARGET_TYPE = False

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

out = 'np/seed1'
os.makedirs(out, exist_ok=True)

print('CTC')

from utils.utils_ctc import get_dataloader
from models.model import Basemodel
from utils.trainer import trainer


print('data loading ...')
path = '/mnt/aoni04/jsakuma/data/sota/'
dataloaders_dict = get_dataloader(path, DENSE_FLAG, ELAN_FLAG, TARGET_TYPE)

net = Basemodel(mode=6, input_size=128, input_img_size=65, input_p_size=64, hidden_size=64)

# 学習済み重みの読み込み
PATH = '/mnt/aoni04/katayama/share/TGNN/baseline/logs/ctc/seed1/202012241122/epoch_30_loss_1.432.pth'
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

from sklearn.metrics import confusion_matrix, classification_report

def msle(y_true, y_pred):
    return (np.log(y_true+1) - np.log(y_pred+1))

timing_error = list()
MSE = []
MSLE = []
missDifference = list()
threshold = 60

Pred = np.array([])
Act = np.array([])
All_Act = np.array([])
y_pred = np.array([])
feature = np.array(dataloaders_dict['val'])

y_true = output['y']

y_prob = output['y_pred']
u_list = output['u']

# 出力の後処理
for i in range(len(y_true)):
    p = y_prob[i]
    
    if y_true[i] < threshold:        
        if y_prob[i] < threshold:
            y_pred = np.append(y_pred, 1)
        else:
            y_pred = np.append(y_pred, 0)
    
    else:
        if y_prob[i] < u_list[i]:
            y_pred = np.append(y_pred, 1)
        else:
            y_pred = np.append(y_pred, 0)

    if y_pred[i] and y_true[i] < 60:
        timing_error.append((p - y_true[i])*50)

        MSLE.append(msle(p*50, y_true[i]*50))
        print(f' 予測:{p} 真値:{y_true[i]} msle {msle(p*50, y_true[i]*50)}')
        Pred = np.append(Pred, p)
        Act = np.append(Act, y_true[i])
        All_Act = np.append(All_Act, y_true[i])


    elif y_true[i] < 60:
        All_Act = np.append(All_Act, y_true[i])
        
    # precision error
    elif y_pred[i]:
#         print(11111)
        missDifference.append(p*50)

y_true = [1 if p < 60 else 0 for p in y_true]    
    
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
print('MAE is {},RMSLE is {}'.format(np.abs(timing_error).mean(),np.mean(MSLE)**0.5))


# 許容度に対する性能計算
y_true = np.array(y_true)

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
hist_ranges = np.arange(0,60,3)
plt.hist(output['y_pred'][y_true==1],label='pred', bins=hist_ranges,histtype='step',linewidth=3.0)
plt.hist(output['y'],label='true', bins=hist_ranges,histtype='step',linewidth=3.0)
plt.legend()
plt.xticks([0,20,40,60],[0,1,2,3])
plt.xlabel('timing [s]')
plt.savefig(out+'/timing_dist.png')