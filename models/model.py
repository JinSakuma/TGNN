
import torch
import torch.nn as nn
import numpy as np
from models.ctr import CTR, CTR_Julius
from models.swt import SWT


class TGNN(nn.Module):
    def __init__(self,
                 mode=0,
                 input_size=128,
                 input_img_size=65,
                 input_p_size=128,
                 hidden_size=64,
                 weight_path='',
                 max_frame=60,
                 ctc=True):
        super(TGNN, self).__init__()
        """
        mode: 0 _ VAD, 1 _ 画像, 2 _ 言語, 3 _ VAD+画像, 4 _ VAD+言語, 5 _ 画像+言語, 6 _ VAD+画像+言語
        """
        self.mode = mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.MSELoss()
        self.max_frame = max_frame
        self.thres1 = 0.8
        self.thres2 = self.get_threshold(max_frame, thres=0.6)  # list型 0~max_frameまでで動的に閾値を持つ
        self.thres_up = 0.5
        self.r = 1.0

        self.input_size = input_size
        self.input_img_size = input_img_size
        self.input_p_size = input_p_size

        if ctc:
            self.ctr = CTR(
                    mode=self.mode,
                    input_size=self.input_size,
                    input_img_size=self.input_img_size,
                    input_p_size=self.input_p_size,
                    )
        else:
            self.ctr = CTR_Julius(
                    mode=self.mode,
                    input_size=self.input_size,
                    input_img_size=self.input_img_size,
                    input_p_size=self.input_p_size,
                    )

        self.swt = SWT(weight_path=weight_path)

        self.hidden_size = hidden_size

    def get_threshold(self, max_frame, thres=0.6):
        """
        thres2を設定する関数
        uが常に1.0の時,max_frameでちょうどthresになるalphaを求め,逆算する
        """
        a = 0
        e = 0.01
        up = 1.0
        min_score = 10000000
        for j in range(100):
            a = a+e
            y_pre = 0
            y = 0
            for i in range(max_frame):
                y = a*up+(1-a)*y_pre
                y_pre = y

            score = abs(y-thres)
            if min_score > score:
                min_score = score
                alpha = a

        up = 1
        a_pre = alpha
        y_pre = 0
        y_list = []
        for i in range(max_frame):
            a_ = up * a_pre + (1-up) * alpha
            y = a_ * up + (1-a_) * y_pre
            a_pre = a_
            y_pre = y
            y_list.append(y)

        return y_list

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
        u_pred_hat = np.asarray([])
        y = np.asarray([])

        if self.mode in [0, 3, 4, 6]:
            hA, hB = self.ctr.calc_voice(xA, xB)

            # SWT 更新するときは使用する予定
            # SWT 更新しないならデータファイルから読み込んできたものを使用
            # up = self.swt(xA, xB)

        if self.mode in [1, 3, 5, 6]:
            hImg = self.ctr.calc_img(img)

        if self.mode in [2, 4, 5, 6]:
            hPA, hPB = self.ctr.calc_lang(PA, PB)

        loss = 0
        l_c = 0
        l_e = 0
        up_pre = 0
        up_max = 0
        cnt = 0
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
            start_frame = 0  # 非発話区間開始フレーム
            u_ = u[i]
            up_ = up[i]
            # u推定値が徐々に下がることを防ぐ
            # 例: up_max=0.8, thres_up=0.5
            #     up_(t-2)=0.8, up_(t-1)=0.6, up_(t)=0.4
            #     -> up_hat(t-2)=0.8, up_hat(t-1)=0.8, up_hat(t)=0.4

            # 非発話区間に入った時にup_max初期化
            if up_pre < self.thres_up and up_ >= self.thres_up:
                up_max = up_
                start_frame = i

            # up_max更新
            if up_max < up_:
                up_max = up_

            up_hat = up_max if up_ >= self.thres_up else up_
#             up_hat = up_

            alpha_ = up_hat * a_pre + (1-up_hat) * a_[i]
            y_ = alpha_ * up_hat + (1-alpha_) * y_pre
            if up_hat < self.thres_up:
                y_ *= 0
            elif start_frame > 0 and i - start_frame > self.max_frame:
                y_ = y_pre

            if label[i] >= self.thres1:
                # uの推定値が閾値以下の場合は最適化から外す
                if up_hat < self.thres_up:
                    l_c = -1
                else:
                    l_c = self.criterion(y_, label[i])
                if self.mode in [2, 4, 5, 6]:
                    self.reset_phoneme()

            # u推定値が切り替わるタイミング
            if up_pre >= self.thres_up and up_hat < self.thres_up:
                # 真値のタイミング
                # u推定値が閾値以下の場合は除外
                if l_c == -1:
                    l_c = 0
                    l_e = 0

                # u推定値が閾値以上の場合
                elif l_c != 0:
                    cnt += 1
                    loss += (l_c * self.r)
                    l_c = 0
                    l_e = 0

                # 正解のタイミングがない場合は閾値を超えていれば最適化
                elif y_pre >= self.thres1:
                    if i-start_frame < self.max_frame:
                        l_e = self.criterion(y_pre, label[i-1]*0+self.thres2[i-start_frame])
                    else:
                        l_e = self.criterion(y_pre, label[i-1]*0+self.thres2[self.max_frame-1])
                        
#                     l_e = self.criterion(y_pre, label[i-1]*0+0.4)
                    cnt += 1
                    loss += l_e
                    l_c = 0
                    l_e = 0
                    
                start_frame = 0

            up_pre = up_hat
            a_pre = alpha_
            y_pre = y_
            alpha = np.append(alpha, alpha_.detach().cpu().numpy())
            u_pred = np.append(u_pred, up_.detach().cpu().numpy())
            u_pred_hat = np.append(u_pred_hat, up_hat.detach().cpu().numpy())
            y = np.append(y, y_.detach().cpu().numpy())
        ##########################

        return {'y': y, 'alpha': alpha, 'u_pred': u_pred, 'u_pred_hat': u_pred_hat, 'loss': loss, 'cnt': cnt}

    def reset_state(self):
        self.ctr.reset_state()
        self.swt.reset_state()

    def back_trancut(self):
        self.ctr.back_trancut()
        self.swt.back_trancut()

    def reset_phoneme(self):
        self.ctr.reset_phoneme()
