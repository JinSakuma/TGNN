import torch
import torch.nn as nn
import numpy as np
from models.ctr import CTR
from models.swt import SWT


class TGNN(nn.Module):
    def __init__(self,
                 mode=0,
                 input_size=128,
                 input_img_size=65,
                 input_p_size=128,
                 hidden_size=64,
                 weight_path=''):
        super(TGNN, self).__init__()
        """
        mode: 0 _ VAD, 1 _ 画像, 2 _ 言語, 3 _ VAD+画像, 4 _ VAD+言語, 5 _ 画像+言語, 6 _ VAD+画像+言語
        """
        self.mode = mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.MSELoss()
        self.thres1 = 0.8
        self.thres2 = 0.4
        self.r = 1.0

        self.input_size = input_size
        self.input_img_size = input_img_size
        self.input_p_size = input_p_size

        self.ctr = CTR(
                mode=self.mode,
                input_size=self.input_size,
                input_img_size=self.input_img_size,
                input_p_size=self.input_p_size,
                )
        self.swt = SWT(weight_path=weight_path)

        self.hidden_size = hidden_size

    def forward(self, batch, y_pre=0, a_pre=0, phase='train'):
        # 1つの入力の時系列の長さ
        self.seq_size = batch['y'].shape[0]

        xA = torch.tensor(batch['voiceA']).to(self.device, dtype=torch.float32)
        xA = xA.unsqueeze(0)
        xB = torch.tensor(batch['voiceB']).to(self.device, dtype=torch.float32)
        xB = xB.unsqueeze(0)
        img = torch.tensor(batch['img']).to(self.device, dtype=torch.float32)
        img = img.unsqueeze(0)
        PA = batch['phonemeA']
        PB = batch['phonemeB']
        label = torch.tensor(batch['y']).to(self.device, dtype=torch.float32)
        u = torch.tensor(batch['u']).to(self.device, dtype=torch.float32)
        up = torch.tensor(batch['u_pred']).to(self.device, dtype=torch.float32)

        # 戻り値 パラメータα(t), 発話期待度y(t)
        alpha = np.asarray([])
        u_pred = np.asarray([])
        y = np.asarray([])

        # print(xA.size())
        # print(xB.size())
        # print(img.size())
        # exit()
        if self.mode in [0, 3, 4, 6]:
            hA, hB = self.ctr.calc_voice(xA, xB)
            up = self.swt(xA, xB)

        if self.mode in [1, 3, 5, 6]:
            hImg = self.ctr.calc_img(img)

        if self.mode in [2, 4, 5, 6]:
            hPA, hPB = self.ctr.calc_lang(PA, PB)

        loss = 0
        l_c = 0
        l_e = 0
        up_pre = 0
        cnt = 0
        calc_flag = False
        # 特徴量のconcat
        if self.mode == 0:
            h = torch.cat([hA, hB], dim=-1)
        elif self.mode == 1:
            h = hImg
        elif self.mode == 2:
            h = torch.cat([hPA, hPB], dim=-1)
        elif self.mode == 3:
            h = torch.cat([hA, hB, hImg], dim=-1)
        elif self.mode == 4:
            h = torch.cat([hA, hB, hPA, hPB], dim=-1)
        elif self.mode == 5:
            h = torch.cat([hImg, hPA, hPB], dim=-1)
        else:
            # print(hA.size())
            # print(hImg.size())
            # print(hPA.size())
            # exit()
            h = torch.cat([hA, hB, hImg, hPA, hPB], dim=-1)

        #  a(t)
        a = self.ctr.calc_alpha(h)
        a_ = a.squeeze()
        #  y(t) ##########
        for i in range(self.seq_size):
            u_ = u[i]
            up_ = up[i]
            alpha_ = up_ * a_pre + (1-up_) * a_[i]
            y_ = alpha_ * up_ + (1-alpha_) * y_pre
            if up_ <= 0.5 or u_ > 1:
                y_ *= 0

            # 真値のタイミング
            if label[i] >= self.thres1:
                l_c = self.criterion(y_, label[i])
#                 self.back_trancut()
                if self.mode in [2, 4, 5, 6]:
                    self.reset_phoneme()

            if y_ >= self.thres1 and not calc_flag:
                l_e = self.criterion(y_, label[i]*0+self.thres2)  # ロスの計算
                calc_flag = True

            # u推定値が切り替わるタイミング
            if (up_pre >= 0.5 and up_ < 0.5) or i == self.seq_size-1:
                if l_c != 0:
                    cnt += 1
                    loss += (l_c * self.r)
                    l_c = 0
                    l_e = 0

                elif l_e != 0:
                    cnt += 1
                    loss += l_e
                    l_c = 0
                    l_e = 0

                calc_flag = False

            up_pre = up_
            a_pre = alpha_
            y_pre = y_
            alpha = np.append(alpha, alpha_.detach().cpu().numpy())
            u_pred = np.append(u_pred, up_.detach().cpu().numpy())
            y = np.append(y, y_.detach().cpu().numpy())
        ##########################

        return {'y': y, 'alpha': alpha, 'u_pred': u_pred, 'loss': loss, 'cnt': cnt}

    def reset_state(self):
        self.ctr.reset_state()
        self.swt.reset_state()

    def back_trancut(self):
        self.ctr.back_trancut()
        self.swt.back_trancut()

    def reset_phoneme(self):
        self.ctr.reset_phoneme()
