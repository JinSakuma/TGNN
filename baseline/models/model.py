
import torch
import torch.nn as nn
import numpy as np


class RMSLELoss(nn.Module):
    """
    Loss function :対数平均2乗誤差を定義
    2020/05/13 : +1e-05 を追加 (lossが0になると微分不可になるため)
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1))+1e-05)


class Basemodel(nn.Module):
    def __init__(self,
                 mode=0,
                 input_size=128,
                 input_img_size=65,
                 input_p_size=128,
                 hidden_size=64,
                 ctc=True):
        super(Basemodel, self).__init__()
        """
        mode: 0 _ VAD, 1 _ 画像, 2 _ 言語, 3 _ VAD+画像, 4 _ VAD+言語, 5 _ 画像+言語, 6 _ VAD+画像+言語
        """
        self.mode = mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = RMSLELoss()
        self.r = 1.0

        self.input_size = input_size
        self.input_img_size = input_img_size
        self.input_p_size = input_p_size

        
        self.hidden_size = hidden_size

        # 音響LSTM
        if self.mode in [0, 3, 4, 6]:
            self.lstm_vad = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True,
            )

        # 画像LSTM
        if self.mode in [1, 3, 5, 6]:
            self.lstm_img = torch.nn.LSTM(
                input_size=input_img_size,
                hidden_size=hidden_size,
                batch_first=True,
            )

        # 言語LSTM
        if self.mode in [2, 4, 5, 6]:
            self.lstm_lng = torch.nn.LSTM(
                input_size=self.input_p_size,
                hidden_size=hidden_size,
                batch_first=True,
            )

        if self.mode == 0 or self.mode == 2:
            self.fc = nn.Linear(hidden_size*2, 1)
        elif self.mode == 1:
            self.fc = nn.Linear(hidden_size, 1)
        elif self.mode == 3 or self.mode == 5:
            self.fc = nn.Linear(hidden_size*3, 1)
        elif self.mode == 4:
            self.fc = nn.Linear(hidden_size*4, 1)
        else:
            self.fc = nn.Linear(hidden_size*5, 1)

        if self.mode in [2, 4, 5, 6]:
            self.prev_hpa = torch.zeros(1, 64).to(self.device)
            self.prev_hpb = torch.zeros(1, 64).to(self.device)
            self.PAD = -1

        self.hidden_size = hidden_size
        self.hidden = None
        self.hiddenA = None
        self.hiddenB = None
        self.hiddenPA = None
        self.hiddenPB = None
        self.hidden_img = None

    def calc_voice(self, xA, xB):
        hA, self.hiddenA = self.lstm_vad(xA, self.hiddenA)
        hB, self.hiddenB = self.lstm_vad(xB, self.hiddenB)
        return hA, hB

    def calc_img(self, img):
        hImg, self.hidden_img = self.lstm_img(img, self.hidden_img)
        return hImg

    def calc_lang(self, PA, PB):
        hpA_list = []
        hpB_list = []
        for i in range(min(len(PA), len(PB))):
            pa = PA[i]
            pb = PB[i]
            if len(pa) == 0:
                hpA = self.prev_hpa
            else:
                if len(pa.shape) == 1:
                    pa = pa.reshape(1, -1)
                pa = torch.FloatTensor(pa).to(self.device)
                pa = pa.unsqueeze(0)
                hpA, self.hiddenPA = self.lstm_lng(pa, self.hiddenPA)
                hpA = hpA[:, -1, :]
            hpA_list.append(hpA)

            if len(pb) == 0:
                hpB = self.prev_hpb
            else:
                if len(pb.shape) == 1:
                    pb = pb.reshape(1, -1)
                pb = torch.FloatTensor(pb).to(self.device)
                pb = pb.unsqueeze(0)
                hpB, self.hiddenPB = self.lstm_lng(pb, self.hiddenPB)
                hpB = hpB[:, -1, :]
            hpB_list.append(hpB)

            self.prev_hpa = hpA
            self.prev_hpb = hpB

        hpa = torch.cat(hpA_list)
        hpb = torch.cat(hpB_list)
        return hpa.unsqueeze(0), hpb.unsqueeze(0)

    def calc_y(self, h):
        y = torch.clamp(self.fc(h), min=0, max=60)
        return y

    def reset_state(self):
        self.hiddenA = None
        self.hiddenB = None
        self.hiddenPA = None
        self.hiddenPB = None
        self.hidden_img = None

    def back_trancut(self):
        if self.hiddenA is not None:
            self.hiddenA = (self.hiddenA[0].detach(), self.hiddenA[1].detach())
            self.hiddenB = (self.hiddenB[0].detach(), self.hiddenB[1].detach())
        if self.hidden_img is not None:
            self.hidden_img = (self.hidden_img[0].detach(), self.hidden_img[1].detach())
        if self.hiddenPA is not None:
            self.hiddenPA = (self.hiddenPA[0].detach(), self.hiddenPA[1].detach())
        if self.hiddenPB is not None:
            self.hiddenPB = (self.hiddenPB[0].detach(), self.hiddenPB[1].detach())

        if self.mode in [2, 4, 5, 6]:
            self.prev_hpa = self.prev_hpa.detach()
            self.prev_hpb = self.prev_hpb.detach()

    def reset_phoneme(self):
        self.prev_hpa = torch.zeros(1, 64).to(self.device)
        self.prev_hpb = torch.zeros(1, 64).to(self.device)

        


    def forward(self, batch, phase='train'):
        # 1つの入力の時系列の長さ
        self.seq_size = batch['voiceA'].shape[0]

        xA = torch.tensor(batch['voiceA']).to(self.device, dtype=torch.float32)
        xA = xA.unsqueeze(0)
        xB = torch.tensor(batch['voiceB']).to(self.device, dtype=torch.float32)
        xB = xB.unsqueeze(0)
        img = torch.tensor(batch['img']).to(self.device, dtype=torch.float32)
        img = img.unsqueeze(0)
        PA = batch['phonemeA']
        PB = batch['phonemeB']
        label = torch.tensor(batch['y']).to(self.device, dtype=torch.float32)


        if self.mode in [0, 3, 4, 6]:
            hA, hB = self.calc_voice(xA, xB)

        if self.mode in [1, 3, 5, 6]:
            hImg = self.calc_img(img)

        if self.mode in [2, 4, 5, 6]:
            hPA, hPB = self.calc_lang(PA, PB)

        loss = 0
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
            h = torch.cat([hA, hB, hImg, hPA, hPB], dim=-1)

        y = self.calc_y(h[:,-1])
        loss = self.criterion(y, label)
        if label < 60:
            loss = (loss * self.r)

        return {'y': y.cpu().data.numpy(), 'loss': loss, 'cnt': cnt}


class MultiTaskmodel(Basemodel):
    def __init__(self,
                 mode=0,
                 input_size=128,
                 input_img_size=65,
                 input_p_size=128,
                 hidden_size=64,
                 ctc=True):
        super().__init__(mode, input_size, input_img_size, input_p_size, hidden_size, ctc)
        self.criterion_act = nn.CrossEntropyLoss()

        if self.mode == 0 or self.mode == 2:
            self.fc2 = nn.Linear(hidden_size*2, 2)
        elif self.mode == 1:
            self.fc2 = nn.Linear(hidden_size, 2)
        elif self.mode == 3 or self.mode == 5:
            self.fc2 = nn.Linear(hidden_size*3, 2)
        elif self.mode == 4:
            self.fc2 = nn.Linear(hidden_size*4, 2)
        else:
            self.fc2 = nn.Linear(hidden_size*5, 2)

    def calc_y(self, h):
        y = torch.clamp(self.fc(h), min=0, max=60)
        y_act = self.fc2(h)
        return y, y_act

    def forward(self, batch, phase='train'):
        # 1つの入力の時系列の長さ
        self.seq_size = batch['voiceA'].shape[0]

        xA = torch.tensor(batch['voiceA']).to(self.device, dtype=torch.float32)
        xA = xA.unsqueeze(0)
        xB = torch.tensor(batch['voiceB']).to(self.device, dtype=torch.float32)
        xB = xB.unsqueeze(0)
        img = torch.tensor(batch['img']).to(self.device, dtype=torch.float32)
        img = img.unsqueeze(0)
        PA = batch['phonemeA']
        PB = batch['phonemeB']
        label = torch.tensor(batch['y']).to(self.device, dtype=torch.float32)
        label_act = torch.tensor(batch['action']).to(self.device, dtype=torch.long)
        label_act = label_act.view(-1)
        if self.mode in [0, 3, 4, 6]:
            hA, hB = self.calc_voice(xA, xB)

        if self.mode in [1, 3, 5, 6]:
            hImg = self.calc_img(img)

        if self.mode in [2, 4, 5, 6]:
            hPA, hPB = self.calc_lang(PA, PB)

        loss = 0
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
            h = torch.cat([hA, hB, hImg, hPA, hPB], dim=-1)

        y, y_act = self.calc_y(h[:,-1])
        loss_timing = self.criterion(y, label)
        #print(y_act.size(),label_act.size(),y.size(),label.size())
        loss_act = self.criterion_act(y_act, label_act)

        loss = loss_timing + 0.5 * loss_act
        if label < 60:             
            loss = (loss * self.r)

        y_act = nn.functional.softmax(y_act, dim=-1).cpu().data.numpy()[:,1]
        return {'y': y.cpu().data.numpy(), 'y_act': y_act, 'loss': loss}
