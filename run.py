import numpy as np
import datetime
import os
import torch
import torch.optim as optim
import argparse
from utils.trainer import trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='/mnt/aoni04/jsakuma/data/sota/')
    parser.add_argument('-l', '--lang', type=str, default='ctc', help='ctc or julius')
#     parser.add_argument('-l', '--lang', type=str, default='julius', help='ctc or julius')
    parser.add_argument('-m', '--mode', type=int, default=6,
                        help='0 _ VAD, 1 _ 画像, 2 _ 言語, 3 _ VAD+画像, 4 _ VAD+言語, 5 _ 画像+言語, 6 _ VAD+画像+言語')
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('--target_type', action='store_true',
                        help='if True, target shape is 3(A,B,unknown), False is 1(A/B)')
    parser.add_argument('-o', '--out', type=str, default='./logs/ctc/p0.4/seed0')
    parser.add_argument('-e', '--epoch', type=int, default=30)
    parser.add_argument('-r', '--resume', type=str, default=True)
    parser.add_argument('--hang', type=str, default=False)

    args = parser.parse_args()

    assert args.lang in args.out, 'args.langと保存先pathを確認!'

    os.makedirs(args.out, exist_ok=True)
    DENSE_FLAG = False
    ELAN_FLAG = True
    TARGET_TYPE = False

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.lang == 'ctc':
        print('CTC')
        from utils.utils_ctc import get_dataloader
        from models.model_ctc import TGNN
    else:
        print('Julius')
        from utils.utils_julius import get_dataloader
        from models.model_julius import TGNN

    now = datetime.datetime.now()
    print('{0:%Y%m%d%H%M}'.format(now))
    out = os.path.join(args.out, '{0:%Y%m%d%H%M}'.format(now))
    os.makedirs(out, exist_ok=True)

    print('data loading ...')
    dataloaders_dict = get_dataloader(args.input, DENSE_FLAG, ELAN_FLAG, TARGET_TYPE)

    if args.lang == 'ctc':
        net = TGNN(mode=args.mode, input_size=128, input_img_size=65, input_p_size=64, hidden_size=64)
    else:
        net = TGNN(mode=args.mode, input_size=128, input_img_size=65, input_p_size=45, hidden_size=64)

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
        if 'fc' in name or 'lstm' in name:
            param.requires_grad = True
            print("勾配計算あり。学習する：", name)
        else:
            param.requires_grad = False
            print("勾配計算なし。学習しない：", name)

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
