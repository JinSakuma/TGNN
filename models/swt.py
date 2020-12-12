import torch
import torch.nn as nn
import numpy as np


def constract_u_t_model(weight_path: str = ''):
    """
    SWT の定義
    Args:
        weight_path (str, optional): [description]. Defaults to ''.
    """
    u_t_model = UnspeechModel()
    
    if weight_path != '':
        print('loaded {}'.format(weight_path))
        u_t_model.load_state_dict(torch.load(weight_path,map_location='cpu'))

    return u_t_model


class UnspeechModel(nn.Module):
    """
    非発話状態の推定network
    """
    def __init__(self, input_size=128, hidden_size=128):
        super(UnspeechModel, self).__init__()

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc2 = nn.Linear(hidden_size, 2)      
        self.softmax = nn.Softmax(dim=-1)

        self.hidden = None
        self.hiddenA = None
        self.hiddenB = None

        self.input_size = input_size
        
    def forward(self, x):
        """
        x .... 入力(spectrogram)
        """
        x = x.view(1, -1, self.input_size)  # 2 -> 3
        h, self.hidden = self.lstm(x, self.hidden)
        h = h.squeeze(0)
        y = self.fc2(h)

        return y

    def reset_state(self):
        self.hidden = None
        self.hiddenA = None
        self.hiddenB = None

    def set_state(self, hidden):
        self.hidden = hidden

    def back_trancut(self):
        if self.hidden is not None:
            self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        if self.hiddenA is not None:
            self.hiddenA = (self.hiddenA[0].detach(), self.hiddenA[1].detach())
        if self.hiddenB is not None:
            self.hiddenB = (self.hiddenB[0].detach(), self.hiddenB[1].detach())

    def get_u_values(self, xA, xB):
        """
        x_A, x_B から u_A, u_B を求めて返す
        """
        h, self.hiddenA = self.lstm(xA, self.hiddenA)
        u_a = self.fc2(h.squeeze(0))

        h, self.hiddenB = self.lstm(xB, self.hiddenB)
        u_b = self.fc2(h.squeeze(0))

        u_a = self.softmax(u_a)
        u_b = self.softmax(u_b)
        u = torch.cat((u_a[:, 1:],u_b[:, 1:]), dim=-1)
        u = torch.min(u_a[:, 1:],u_b[:, 1:])

        return u, u_a, u_b


class SWT(nn.Module):
    """
    SWT
    非発話度 u(t) を推定
    """
    def __init__(self, 
                 weight_path='',
                 ):
        super(SWT, self).__init__()

        self.u_t_model = constract_u_t_model(weight_path)
        self.criterion_u = nn.CrossEntropyLoss()

        self.swt_loss = 0.0
        self.swt_total_loss = 0.0

    def forward(self,xA, xB):
        u, self.uA, self.uB = self.u_t_model.get_u_values(xA, xB)
        return u

    def initialine(self):
        """
        各epochの開始時に呼び出される
        """
        self.swt_loss = 0.0
        self.swt_total_loss = 0.0

    def reset_state(self):
        """
        会話毎の境目で呼び出される
        LSTM のstateをreset
        """
        self.u_t_model.reset_state()

    def back_trancut(self):
        """
        BP後に呼び出される
        LSTMのstateの逆伝搬を切り取る
        (順伝搬に対するstateは保持される)
        """
        self.u_t_model.back_trancut()
        self.swt_loss = 0.0

    def compute_loss(self, yA, yB):
        """
        SWT の出力部分で損失を計算
        """
        tA = torch.clamp(yA, 0, 1)
        tB = torch.clamp(yB, 0, 1)
        loss = self.criterion_u(self.uA, tA) + self.criterion_u(self.uB, tB)
        self.swt_loss += loss
        self.swt_total_loss += loss
