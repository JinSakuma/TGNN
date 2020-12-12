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


    def forward(self, x, PA, PB, u, up, label, y_pre=0, a_pre=0, phase='train'):
        # 1つの入力の時系列の長さ
        self.seq_size = x.shape[0]

        # 戻り値 パラメータα(t), 発話期待度y(t)
        alpha_0 = np.asarray([])
        alpha = np.asarray([])
        y = np.asarray([])

        # 音響&画像特徴量
        x = x.unsqueeze(0)
        xA = x[:, :, self.input_img_size+self.input_size:self.input_img_size+self.input_size*2]
        xB = x[:, :, self.input_img_size:self.input_img_size+self.input_size]
        img = x[:, :, :self.input_img_size]

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
        loss_ = 0
        l_c = 0
        l_e = 0
        u_pre = 0
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
        alpha_0 = np.append(alpha_0, a_.detach().cpu().numpy())  
        #y(t) ##########
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
            
            if y_ >= self.thres1 and calc_flag==False:
                l_e = self.criterion(y_, label[i]*0+self.thres2)  # ロスの計算
                calc_flag = True

            # u推定値が切り替わるタイミング
            if (up_pre >= 0.5 and up_ < 0.5) or i == self.seq_size-1:
                if l_c != 0:
                    cnt += 1
                    loss += (l_c * self.r)
                    loss_ += float(l_c.item() * self.r)
                    l_c = 0
                    l_e = 0

                elif l_e != 0:
                    cnt += 1
                    loss += l_e
                    loss_ += float(l_e.item())
                    l_c = 0
                    l_e = 0
                
                calc_flag = False

            up_pre = up_
            u_pre = u_
            a_pre = alpha_
            y_pre = y_
            alpha = np.append(alpha, alpha_.detach().cpu().numpy())
            y = np.append(y, y_.detach().cpu().numpy())
        ##########################

        return y, alpha, alpha_0, up, loss, loss_, cnt

    def reset_state(self):
        self.ctr.reset_state()
        self.swt.reset_state()

    def back_trancut(self):
        self.ctr.back_trancut()
        self.swt.back_trancut()

    def reset_phoneme(self):
        self.ctr.reset_phoneme()




