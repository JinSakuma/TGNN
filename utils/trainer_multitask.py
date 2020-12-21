import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.eval import quantitative_evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train(net, mode, dataloaders_dict,
          device, optimizer, scheduler
          ):

    #  dataloaders_dict['train'].on_epoch_end()
    net.train()  # モデルを訓練モードに
#     scheduler.step()
#     print('lr:{}'.format(scheduler.get_lr()[0]))

    epoch_loss, epoch_alpha_loss, epoch_action_loss = 0, 0, 0
    train_cnt, train_act_cnt = 0, 0
    for batch in tqdm(dataloaders_dict['train']):
        out, a = np.zeros(5), np.zeros(5)
        net.reset_state()
        if mode == 2 or mode >= 4:
            net.reset_phoneme()

        for i in range(len(batch[0])):
            output_dict = net(batch[0][i], out[-1], a[-1])

            loss = output_dict['loss']
            action_loss = output_dict['loss_act']
            alpha_loss = output_dict['loss_y']
            if loss != 0 and loss != -1:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                net.back_trancut()
                loss = loss.item()

            epoch_action_loss += float(action_loss)
            epoch_alpha_loss += float(alpha_loss)
            train_cnt += output_dict['cnt']
            train_act_cnt += output_dict['action_cnt']
            loss = 0

    epoch_action_loss = epoch_action_loss / train_act_cnt
    epoch_alpha_loss = epoch_alpha_loss / train_cnt
    epoch_loss = epoch_action_loss + epoch_alpha_loss * net.r
    output_dict = {}

    return epoch_loss, epoch_action_loss, epoch_alpha_loss


def val(net, mode, dataloaders_dict,
        device, optimizer, scheduler,
        epoch, output='./', resume=False
        ):

    #  dataloaders_dict['val'].on_epoch_end()
    net.eval()  # モデルを訓練モードに
    epoch_loss, epoch_alpha_loss, epoch_action_loss = 0, 0, 0
    train_cnt, train_act_cnt = 0, 0
    threshold = 0.8
    a_pred = np.array([])
    u_true, u_pred, u_pred_hat = np.array([]), np.array([]), np.array([])
    y_true, y_pred = np.array([]), np.array([])
    act_true, act_pred = np.array([]), np.array([])
    act_true_list, act_pred_list = np.array([]), np.array([])

    with torch.no_grad():
        for batch in tqdm(dataloaders_dict['val']):
            out, a = np.zeros(5), np.zeros(5)
            net.reset_state()
            if mode == 2 or mode >= 4:
                    net.reset_phoneme()

            for i in range(len(batch[0])):
                output_dict = net(batch[0][i], out[-1], a[-1], phase='val')

                a_pred = np.append(a_pred, output_dict['alpha'])
                u_true = np.append(u_true, batch[0][i]['u'])
                u_pred = np.append(u_pred, output_dict['u_pred'])
                u_pred_hat = np.append(u_pred_hat, output_dict['u_pred_hat'])
                y_true = np.append(y_true, batch[0][i]['y'])
                y_pred = np.append(y_pred, output_dict['y'])
                act_pred = np.append(act_pred, output_dict['action'])
                act_true = np.append(act_true, batch[0][i]['action'])
                act_true_list = np.append(act_true_list, output_dict['action_label'])
                act_pred_list = np.append(act_pred_list, output_dict['action_pred'])

                loss = output_dict['loss']
                action_loss = output_dict['loss_act']
                alpha_loss = output_dict['loss_y']
                if loss != 0 and loss != -1:
                    net.back_trancut()
                    loss = loss.item()

                epoch_action_loss += float(action_loss)
                epoch_alpha_loss += float(alpha_loss)
                train_cnt += output_dict['cnt']
                train_act_cnt += output_dict['action_cnt']
                loss = 0

        epoch_action_loss = epoch_action_loss / train_act_cnt
        epoch_alpha_loss = epoch_alpha_loss / train_cnt
        epoch_loss = epoch_action_loss + epoch_alpha_loss * net.r

        if resume:
            fig = plt.figure(figsize=(20, 8))
            plt.rcParams["font.size"] = 18
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)

            ax1.plot([threshold]*300, color='black', linestyle='dashed')
            ax1.plot(u_true[:300], label='u_true', color='g', linewidth=3.0)
            ax1.plot(u_pred[:300], label='u_pred', color='g', linewidth=2.0)
            ax1.plot(u_pred_hat[:300], label='u_pred_hat', color='m', linewidth=2.0)
            ax1.legend()
            # plt.fill_between(range(300), u_pred[:300], color='g', alpha=0.3)
            ax2.plot([threshold]*300, color='black', linestyle='dashed')
            ax2.plot(y_pred[:300], label='predict', linewidth=3.0)
            ax2.plot(a_pred[:300], label='a_t', color='r', linewidth=3.0)
            ax2.fill_between(range(300), a_pred[:300], color='r', alpha=0.3)
            ax2.plot(y_true[:300], label='true label', linewidth=4.0, color='b')
            ax2.legend()
            plt.savefig(os.path.join(output, 'result_{}_loss_{:.3f}.png'.format(epoch+1, epoch_loss)))
            plt.close()

            precision, recall, f1, Distance, LogDistance, score = quantitative_evaluation(epoch+1, y_true, y_pred, u_true, threshold=threshold, resume=True, output=output)
            torch.save(net.state_dict(), os.path.join(output, 'epoch_{}_loss_{:.4f}_score{:.3f}.pth'.format(epoch+1, epoch_loss, score)))
            print('-------------')

            fo = open(os.path.join(output, 'eval_report.txt'), 'a')
            print(accuracy_score(act_true_list, act_pred_list), file=fo)
            print(confusion_matrix(act_true_list, act_pred_list), file=fo)
            print(classification_report(act_true_list, act_pred_list), file=fo)
            print('----------------------------------------------', file=fo)

    output_dict = {}

    return epoch_loss, epoch_action_loss, epoch_alpha_loss


def trainer(net,
            mode,
            dataloaders_dict,
            optimizer, scheduler,
            num_epochs=10,
            output='./',
            resume=False,
            ):

    os.makedirs(output, exist_ok=True)
    Loss = {'train': [], 'val': []}
    action_Loss = {'train': [], 'val': []}
    alpha_Loss = {'train': [], 'val': []}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using', device)
    net.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        for phase in ['train', 'val']:
            print(phase)
            if phase == 'train':
                epoch_loss, epoch_action_loss, epoch_alpha_loss = train(net, mode, dataloaders_dict, device, optimizer, scheduler)
            else:
                epoch_loss, epoch_action_loss, epoch_alpha_loss = val(net, mode, dataloaders_dict, device,
                                                                      optimizer, scheduler, epoch, output, resume)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            Loss[phase].append(epoch_loss)
            action_Loss[phase].append(epoch_action_loss)
            alpha_Loss[phase].append(epoch_alpha_loss)

    if resume:
        plt.figure(figsize=(15, 4))
        plt.rcParams["font.size"] = 15
        plt.plot(Loss['val'], label='val')
        plt.plot(Loss['train'], label='train')
        plt.legend()
        plt.savefig(os.path.join(output, 'history.png'))
        plt.close()

        plt.figure(figsize=(15, 4))
        plt.rcParams["font.size"] = 15
        plt.plot(action_Loss['val'], label='val')
        plt.plot(action_Loss['train'], label='train')
        plt.legend()
        plt.savefig(os.path.join(output, 'history_action.png'))
        plt.close()

        plt.figure(figsize=(15, 4))
        plt.rcParams["font.size"] = 15
        plt.plot(alpha_Loss['val'], label='val')
        plt.plot(alpha_Loss['train'], label='train')
        plt.legend()
        plt.savefig(os.path.join(output, 'history_alpha.png'))
        plt.close()
