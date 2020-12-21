import torch
import torch.nn as nn


class CTR(nn.Module):
    """
    CTR
    応答速度 alpha(t) を推定
    """
    def __init__(self,
                 mode=0,
                 input_size=128,
                 input_img_size=65,
                 input_p_size=128,
                 hidden_size=64
                 ):
        super(CTR, self).__init__()
        self.mode = mode

        self.input_size = input_size
        self.input_img_size = input_img_size
        self.input_p_size = input_p_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    def calc_alpha(self, h):
        alpha = torch.sigmoid(self.fc(h))
        return alpha

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


class CTR_Julius(CTR):
    """
    CTR
    応答速度 alpha(t) を推定
    """
    def __init__(self,
                 mode=0,
                 input_size=128,
                 input_img_size=65,
                 input_p_size=45,
                 hidden_size=64
                 ):
        super().__init__(mode, input_size, input_img_size, input_p_size, hidden_size)

        self.embedding_size = 30

        # 言語LSTM
        if self.mode in [2, 4, 5, 6]:
            # 埋め込み層
            self.embedding = nn.Embedding(self.input_p_size, self.embedding_size)
            self.lstm_lng = torch.nn.LSTM(
                input_size=self.embedding_size,
                hidden_size=hidden_size,
                batch_first=True,
            )

    def calc_lang(self, PA, PB):
        hpA_list = []
        hpB_list = []
        for i in range(min(len(PA), len(PB))):
            pa = PA[i]
            pb = PB[i]
            if pa == self.PAD:
                hpA = self.prev_hpa
            else:
                pA = torch.tensor(pa).to(self.device, dtype=torch.long)
                emb_pA = self.embedding(pA)
                emb_pA = emb_pA.unsqueeze(0)
                hpA, self.hiddenPA = self.lstm_lng(emb_pA, self.hiddenPA)
                hpA = hpA[:, -1, :]
            hpA_list.append(hpA)

            if pb == self.PAD:
                hpB = self.prev_hpb
            else:
                pB = torch.tensor(pb).to(self.device, dtype=torch.long)
                emb_pB = self.embedding(pB)
                emb_pB = emb_pB.unsqueeze(0)
                hpB, self.hiddenPA = self.lstm_lng(emb_pB, self.hiddenPB)
                hpB = hpB[:, -1, :]
            hpB_list.append(hpB)

            self.prev_hpa = hpA
            self.prev_hpb = hpB

        hpa = torch.cat(hpA_list)
        hpb = torch.cat(hpB_list)
        return hpa.unsqueeze(0), hpb.unsqueeze(0)


class CTR_Multitask(CTR):
    """
    CTR
    応答速度 alpha(t) を推定
    """
    def __init__(self,
                 mode=0,
                 input_size=128,
                 input_img_size=65,
                 input_p_size=45,
                 hidden_size=64,
                 num_cls=2):
        super().__init__(mode, input_size, input_img_size, input_p_size, hidden_size)

        self.num_cls = num_cls

        if self.mode == 0 or self.mode == 2:
            self.fc_act_1 = nn.Linear(hidden_size*2, hidden_size)
            self.fc_act_2 = nn.Linear(hidden_size, self.num_cls)
        elif self.mode == 1:
            # self.fc_act_1 = nn.Linear(hidden_size*3, hidden_size)
            self.fc_act_2 = nn.Linear(hidden_size, self.num_cls)
        elif self.mode == 3 or self.mode == 5:
            self.fc_act_1 = nn.Linear(hidden_size*3, hidden_size)
            self.fc_act_2 = nn.Linear(hidden_size, self.num_cls)
        elif self.mode == 4:
            self.fc_act_1 = nn.Linear(hidden_size*4, hidden_size)
            self.fc_act_2 = nn.Linear(hidden_size, self.num_cls)
        else:
            self.fc_act_1 = nn.Linear(hidden_size*5, hidden_size)
            self.fc_act_2 = nn.Linear(hidden_size, self.num_cls)

    def predict_action(self, h):

        if self.mode != 1:
            cls = self.fc_act_1(h)
            cls = self.fc_act_2(cls)
        else:
            cls = self.fc_act_2(h)

        return cls


class CTR_Multitask_Julius(CTR_Julius):
    """
    CTR
    応答速度 alpha(t) を推定
    """
    def __init__(self,
                 mode=0,
                 input_size=128,
                 input_img_size=65,
                 input_p_size=45,
                 hidden_size=64,
                 num_cls=2
                 ):
        super().__init__(mode, input_size, input_img_size, input_p_size, hidden_size)

        self.num_cls = num_cls

        if self.mode == 0 or self.mode == 2:
            self.fc_act_1 = nn.Linear(hidden_size*2, hidden_size)
            self.fc_act_2 = nn.Linear(hidden_size, self.num_cls)
        elif self.mode == 1:
            # self.fc_act_1 = nn.Linear(hidden_size*3, hidden_size)
            self.fc_act_2 = nn.Linear(hidden_size, self.num_cls)
        elif self.mode == 3 or self.mode == 5:
            self.fc_act_1 = nn.Linear(hidden_size*3, hidden_size)
            self.fc_act_2 = nn.Linear(hidden_size, self.num_cls)
        elif self.mode == 4:
            self.fc_act_1 = nn.Linear(hidden_size*4, hidden_size)
            self.fc_act_2 = nn.Linear(hidden_size, self.num_cls)
        else:
            self.fc_act_1 = nn.Linear(hidden_size*5, hidden_size)
            self.fc_act_2 = nn.Linear(hidden_size, self.num_cls)

    def predict_action(self, h):

        if self.mode != 1:
            cls = self.fc_act_1(h)
            cls = self.fc_act_2(cls)
        else:
            cls = self.fc_act_2(h)

        return cls
