import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.eval import quantitative_evaluation
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def train(net, mode, dataloaders_dict,
          device, optimizer, scheduler
          ):

    #  dataloaders_dict['train'].on_epoch_end()
    net.train()  # モデルを訓練モードに
#     scheduler.step()
#     print('lr:{}'.format(scheduler.get_lr()[0]))

    epoch_loss = 0.0  # epochの損失和
    train_cnt = 0
    for batch in tqdm(dataloaders_dict['train']):
        out, a = np.zeros(5), np.zeros(5)
        net.reset_state()
        if mode == 2 or mode >= 4:
            net.reset_phoneme()

        for i in range(len(batch[0])):
            output_dict = net(batch[0][i], out[-1], a[-1])

            loss = output_dict['loss']
            if loss != 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                net.back_trancut()
                loss = loss.item()

            epoch_loss += loss
            loss = 0
            train_cnt += output_dict['cnt']

    epoch_loss = epoch_loss / train_cnt
    output_dict = {}

    return epoch_loss


def val(net, mode, dataloaders_dict,
        device, optimizer, scheduler,
        epoch, output='./', resume=False
        ):

    #  dataloaders_dict['val'].on_epoch_end()
    net.eval()  # モデルを訓練モードに
    epoch_loss = 0.0  # epochの損失和
    train_cnt = 0
    threshold = 0.8
    a_pred = np.array([])
    u_true, u_pred = np.array([]), np.array([])
    y_true, y_pred = np.array([]), np.array([])

    with torch.no_grad():
        for batch in tqdm(dataloaders_dict['val']):
            out, a = np.zeros(5), np.zeros(5)
            net.reset_state()
            if mode == 2 or mode >= 4:
                    net.reset_phoneme()

            for i in range(len(batch[0])):
                output_dict = net(batch[0][i], out[-1], a[-1], phase='val')

                a_pred = np.append(a_pred, output_dict['alpha'])
                u_true = np.append(u_true, batch[0][i]['U'])
                u_pred = np.append(u_pred, output_dict['u_pred'])
                y_true = np.append(y_true, batch[0][i]['y'])
                y_pred = np.append(y_pred, output_dict['y'])

                loss = output_dict['loss']
                if loss != 0:
                    net.back_trancut()
                    loss = loss.item()

                epoch_loss += loss
                loss = 0
                train_cnt += output_dict['cnt']

        epoch_loss = epoch_loss / train_cnt
        if resume:
            torch.save(net.state_dict(), os.path.join(output, 'epoch_{}_loss_{:.3f}.pth'.format(epoch+1, epoch_loss)))
            plt.figure(figsize=(20, 4))
            plt.rcParams["font.size"] = 18
            plt.plot([threshold]*300, color='black', linestyle='dashed')
            plt.plot(y_pred[:300], label='predict', linewidth=3.0)
            plt.plot(u_true[:300], label='u_true', color='g', linewidth=3.0)
            plt.plot(u_pred[:300], label='u_t', color='g', linewidth=2.0)
            plt.fill_between(range(300), u_pred[:300], color='g', alpha=0.3)
            plt.plot(a_pred[:300], label='a_t', color='r', linewidth=3.0)
            plt.fill_between(range(300), a_pred[:300], color='r', alpha=0.3)
            plt.plot(y_true[:300], label='true label', linewidth=4.0, color='m')
            plt.legend()
            plt.savefig(os.path.join(output, 'result_{}_loss_{:.3f}.png'.format(epoch+1, epoch_loss)))

            precision, recall, f1, Distance, LogDistance = quantitative_evaluation(epoch+1, y_true, y_pred, u_true, threshold=threshold, resume=True, output=output)
            plt.close()
            print('-------------')

    return epoch_loss


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using', device)
    net.to(device)

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
