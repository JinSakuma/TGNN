import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.eval import quantitative_evaluation
from utils.utils import add_hang_over

def train(net, mode, dataloaders_dict,
          device, optimizer, scheduler
          ):

    net.train()  # モデルを訓練モードに

    epoch_loss = 0.0  # epochの損失和
    total_loss = 0.0
    train_cnt = 0
    for batch in tqdm(dataloaders_dict['train']):
        back_count = 0

        net.reset_state()
        if mode == 2 or mode >= 4:
            net.reset_phoneme()

        for i in range(len(batch[0])):
            output_dict = net(batch[0][i])

            loss = output_dict['loss']
            if loss != 0:
                total_loss += loss
                train_cnt += output_dict['cnt']
                back_count += 1

                loss = loss.item()
                epoch_loss += loss

            if back_count % 8 == 7:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                net.back_trancut()

                total_loss = 0.0
                back_count = 0


    epoch_loss = epoch_loss / train_cnt
    output_dict = {}

    return epoch_loss


def val(net, mode, dataloaders_dict,
        device, optimizer, scheduler,
        epoch, output='./', resume=False, only_eval=False
        ):

    net.eval()  # モデルを訓練モードに
    epoch_loss = 0.0  # epochの損失和
    train_cnt = 0
    threshold = 0.5
    a_pred = np.array([])
    u_true, silence_list = np.array([]), np.array([])
    y_true, y_pred = np.array([]), np.array([])

    with torch.no_grad():
        for batch in tqdm(dataloaders_dict['val']):
            net.reset_state()
            if mode == 2 or mode >= 4:
                    net.reset_phoneme()

            for i in range(len(batch[0])):
                output_dict = net(batch[0][i], phase='val')

                u_true = np.append(u_true, batch[0][i]['u'])

                y_true = np.append(y_true, batch[0][i]['y'])
                y_pred = np.append(y_pred, output_dict['y'])
                silence_list = np.append(silence_list, output_dict['silence'])

                loss = output_dict['loss']
                if loss != 0:
                    loss = loss.item()

                epoch_loss += loss
                train_cnt += output_dict['cnt']

        epoch_loss = epoch_loss / train_cnt

        if only_eval:
            return {
                'y':y_true,
                'y_pred': y_pred,
                'u': u_true,
                'silence': silence_list
            }

        if resume:
            # 評価用に後処理
            y_true2 = y_true[1:] - y_true[:-1]
            y_true2 = np.maximum(y_true2, 0)
            precision, recall, f1, Distance, LogDistance, _ = quantitative_evaluation(
                                                            epoch+1,y_true2, y_pred, u_true, threshold=threshold, 
                                                            resume=True, output=output, eval_flg=False, only_VAD_ON=True, HO=0)

            torch.save(net.state_dict(), os.path.join(output, 'epoch_{}_loss_{:.4f}_score{:.3f}.pth'.format(epoch+1, epoch_loss, f1)))
            print('-------------')

            fig = plt.figure(figsize=(20, 8))
            plt.rcParams["font.size"] = 18
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)

            ax1.plot([threshold]*300, color='black', linestyle='dashed')
            ax1.plot(u_true[:300], label='u_true', color='g', linewidth=3.0)
            silence_list = silence_list.reshape(-1)
            ax1.plot(silence_list[:300] / 60, label='silence', color='g', linewidth=3.0)
            ax1.legend()
            # ax2.plot([threshold]*300, color='black', linestyle='dashed')
            ax2.plot(y_pred[:300], label='predict', linewidth=3.0)
            ax2.plot(y_true[:300], label='true label', linewidth=4.0, color='b')
            ax2.legend()
            plt.savefig(os.path.join(output, 'result_{}_loss_{:.3f}.png'.format(epoch+1, epoch_loss)))
            plt.close()


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
                    epoch_loss = train(net, mode, dataloaders_dict, device, optimizer, scheduler)
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