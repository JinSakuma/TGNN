import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from utils.eval import quantitative_evaluation


def train(net, mode, dataloaders_dict,
          device, optimizer, scheduler,
          epoch=10, output='./'
          ):
    net.train()  # モデルを訓練モードに

    epoch_loss = 0.0  # epochの損失和
    train_cnt = 0
    y, y_pred = np.array([]), np.array([])
    y_prob, u_list = np.array([]), np.array([])
    for batch in tqdm(dataloaders_dict['train']):
        net.reset_state()
        if mode == 2 or mode >= 4:
            net.reset_phoneme()

        for i in range(len(batch[0])):
            output_dict = net(batch[0][i])
            y = np.append(y, batch[0][i]['y'])
            y_prob = np.append(y_prob, output_dict['y'])

            u_list = np.append(u_list ,batch[0][i]['u'])
            loss = output_dict['loss']
            if loss != 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                net.back_trancut()
                loss = loss.item()

            epoch_loss += loss
            loss = 0
            train_cnt += 1

    epoch_loss = epoch_loss / train_cnt
    
    y_true = [1 if p < 60 else 0 for p in y]
    y_pred = np.zeros(len(y_true))

    """
    memo: 予測値が６０未満なら turn交代(1)と予測
          (非発話区間の長さ) < (予測値) なら turn継続(0)と予測
    """
    timing_err = []
    for j in range(len(y_true)):
        if y_true[j]:
            if y_prob[j] < 60:
                y_pred[j] = 1
                timing_err.append((y_prob[j] - y[j]) * 50)
        else:
            if y_prob[j] < u_list[j]:
                y_pred[j] = 1

    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    print('MAE: {} ms'.format(np.abs(timing_err).mean()))

    fo = open(os.path.join(output, 'eval_report.txt'), 'a')
    print('Epoch {} Training Phase'.format(epoch+1), file=fo)
    print(classification_report(y_true, y_pred),file=fo)
    print('MAE: {} ms'.format(np.abs(timing_err).mean()), file=fo)
    fo.close()

    output_dict = {}

    return epoch_loss


def val(net, mode, dataloaders_dict,
        device, optimizer, scheduler,
        epoch=10, output='./', resume=False,
        only_eval=False
        ):

    net.eval()  # モデルを訓練モードに
    epoch_loss = 0.0  # epochの損失和
    train_cnt = 0
    y, y_pred = np.array([]), np.array([])
    y_prob, u_list = np.array([]), np.array([])
    with torch.no_grad():
        for batch in tqdm(dataloaders_dict['val']):
            net.reset_state()
            if mode == 2 or mode >= 4:
                net.reset_phoneme()

            for i in range(len(batch[0])):
                output_dict = net(batch[0][i], phase='val')

                y = np.append(y, batch[0][i]['y'])
                y_prob = np.append(y_prob, output_dict['y'])

                u_list = np.append(u_list ,batch[0][i]['u'])

                loss = output_dict['loss']
                if loss != 0 and loss != -1:
                    loss = loss.item()

                epoch_loss += loss
                loss = 0
                train_cnt += 1

        epoch_loss = epoch_loss / train_cnt

    if only_eval:
        return {
            'y':y,
            'y_pred':y_prob,
            'u': u_list,
            'loss':epoch_loss
                }

    torch.save(net.state_dict(), os.path.join(output, 'epoch_{}_loss_{:.3f}.pth'.format(epoch+1, epoch_loss)))

    y_true = [1 if p < 60 else 0 for p in y]
    y_pred = np.zeros(len(y_true))

    """
    memo: 予測値が６０未満なら turn交代(1)と予測
          (非発話区間の長さ) < (予測値) なら turn継続(0)と予測
    """
    timing_err = []
    for j in range(len(y_true)):
        if y_true[j] < 60:
            if y_prob[j] < 60:
                y_pred[j] = 1
                timing_err.append((y_prob[j] - y[j]) * 50)
        else:
            if y_prob[j] < u_list[j]:
                y_pred[j] = 1

    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    print('MAE: {} ms'.format(np.abs(timing_err).mean()))

    fo = open(os.path.join(output, 'eval_report.txt'), 'a')
    print('Epoch {}'.format(epoch+1), file=fo)
    print(classification_report(y_true, y_pred),file=fo)
    print('MAE: {} ms'.format(np.abs(timing_err).mean()), file=fo)
    fo.close()

    #保存
    # hist_ranges = np.arange(0,60,3)
    # plt.hist(y_prob[y_true==1],label='pred', bins=hist_ranges,histtype='step',linewidth=3.0)
    # plt.hist(y[y_true==1],label='true', bins=hist_ranges,histtype='step',linewidth=3.0)
    # plt.legend()
    # plt.xticks([0,20,40,60],[0,1,2,3])
    # plt.xlabel('timing [s]')
    # plt.savefig(os.path.join(output, 'epoch_{}_loss_{:.3f}.png'.format(epoch+1, epoch_loss)))

    return epoch_loss


def trainer(net,
            mode,
            dataloaders_dict,
            optimizer, scheduler,
            num_epochs=10,
            output='./',
            resume=False,
            only_eval=False
            ):

    os.makedirs(output, exist_ok=True)
    Loss = {'train': [], 'val': []}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using', device)
    net.to(device)

    if not only_eval:
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-------------')

            for phase in ['train', 'val']:
                print(phase)

                if phase == 'train':
                    epoch_loss = train(net, mode, dataloaders_dict, device, optimizer, scheduler,
                                        epoch, output)
                else:
                    epoch_loss = val(net, mode, dataloaders_dict, device,
                                    optimizer, scheduler, epoch, output, resume)

                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                Loss[phase].append(epoch_loss)

        if resume:
            plt.figure(figsize=(15, 4))
            plt.rcParams["font.size"] = 15
            plt.plot(Loss['val'], label='val')
            plt.plot(Loss['train'], label='train')
            plt.legend()
            plt.savefig(os.path.join(output, 'history.png'))
    else:
        output = val(net, mode, dataloaders_dict, device,
                    optimizer, scheduler, 0, output, resume, only_eval=only_eval)

        return output