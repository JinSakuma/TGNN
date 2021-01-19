import numpy as np
import datetime
import os
import torch
import torch.optim as optim
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='/mnt/aoni04/jsakuma/data/sota/')
    parser.add_argument('-l', '--lang', type=str, default='ctc', help='ctc or julius')
#     parser.add_argument('-l', '--lang', type=str, default='julius', help='ctc or julius')
    parser.add_argument('-m', '--mode', type=int, default=6,
                        help='0 _ VAD, 1 _ 画像, 2 _ 言語, 3 _ VAD+画像, 4 _ VAD+言語, 5 _ 画像+言語, 6 _ VAD+画像+言語')
    parser.add_argument('-t', '--task', type=bool, default=False,
                        help='true: multitask, false: singletask')
    parser.add_argument('-s', '--seed', type=int, default=2)
    parser.add_argument('--target_type', action='store_true',
                        help='if True, target shape is 3(A,B,unknown), False is 1(A/B)')
    parser.add_argument('-o', '--out', type=str, default='./logs/ctc/0112_not_add/vip0.4/seed2')
    parser.add_argument('-e', '--epoch', type=int, default=30)
    parser.add_argument('-r', '--resume', type=str, default=True)
    parser.add_argument('--hang', type=str, default=False)
    parser.add_argument('--gpuid', type=int, default=1)
    parser.add_argument('--weight', type=str,
                        default='/mnt/aoni04/katayama/share/SPEC/epoch_20_acc0.887_loss0.266_ut_train.pth')

    args = parser.parse_args()

    assert args.lang in args.out, 'args.langと保存先pathを確認!'

    os.makedirs(args.out, exist_ok=True)
    DENSE_FLAG = False
    ELAN_FLAG = True
    TARGET_TYPE = False

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

    now = datetime.datetime.now()
    print('{0:%Y%m%d%H%M}'.format(now))
    out = os.path.join(args.out, '{0:%Y%m%d%H%M}'.format(now))
    os.makedirs(out, exist_ok=True)

    if args.task:
        from models.model_multitask import TGNN
        from utils.trainer_multitask import trainer
    else:
        from models.model import TGNN
        from utils.trainer import trainer

    # モデル設定
    input_size = 128
    input_img_size = 65
    hidden_size = 64
    if args.lang == 'ctc':
        print('CTC')
        from utils.utils_ctc import get_dataloader
        input_p_size = 64
        ctc_flg = True
    else:
        print('Julius')
        from utils.utils_julius import get_dataloader
        input_p_size = 45
        ctc_flg = False

    print('data loading ...')
    dataloaders_dict = get_dataloader(args.input, DENSE_FLAG, ELAN_FLAG, TARGET_TYPE)

    net = TGNN(mode=args.mode,
               input_size=input_size,
               input_img_size=input_img_size,
               input_p_size=input_p_size,
               hidden_size=hidden_size,
               weight_path=args.weight,
               ctc=ctc_flg)

    if args.mode == 0:
        print("音響")
    elif args.mode == 1:
        print("画像")
    elif args.mode == 2:
        print("音素")
    elif args.mode == 3:
        print("音響+画像")
    elif args.mode == 4:
        print("音響+音素")
    elif args.mode == 5:
        print("画像+音素")
    else:
        print("音響+画像+音素")

    print('Model :', net.__class__.__name__)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    for name, param in net.named_parameters():
        if 'swt' in name:
            param.requires_grad = False
            print("勾配計算なし。学習しない：", name)
        else:
            param.requires_grad = True
            print("勾配計算あり。学習する：", name)

    print('train data is ', len(dataloaders_dict['train']))
    print('test data is ', len(dataloaders_dict['val']))

    trainer(
        net=net,
        mode=args.mode,
        dataloaders_dict=dataloaders_dict,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epoch,
        output=out,
        resume=args.resume,
        )


if __name__ == '__main__':
    main()
