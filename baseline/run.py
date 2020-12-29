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
    parser.add_argument('-m', '--mode', type=int, default=6,
                        help='0 _ VAD, 1 _ 画像, 2 _ 言語, 3 _ VAD+画像, 4 _ VAD+言語, 5 _ 画像+言語, 6 _ VAD+画像+言語')
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('--target_type', action='store_true',
                        help='if True, target shape is 3(A,B,unknown), False is 1(A/B)')
    parser.add_argument('-o', '--out', type=str, default='./logs/ctc')
    parser.add_argument('-e', '--epoch', type=int, default=30)
    parser.add_argument('-r', '--resume', type=str, default=True)
    parser.add_argument('--hang', type=str, default=False)
    parser.add_argument('-t', '--task', type=bool, default=False,
                        help='true: multitask, false: singletask')
    parser.add_argument('--gpuid', type=int, default=0)
    
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

    out = os.path.join(args.out, 'seed{}'.format(seed))
    now = datetime.datetime.now()
    print('{0:%Y%m%d%H%M}'.format(now))
    out = os.path.join(out, '{0:%Y%m%d%H%M}'.format(now))
    os.makedirs(out, exist_ok=True)

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

    if args.task:
        from utils.trainer_multitask import trainer
        from models.model import MultiTaskmodel
        net = MultiTaskmodel(mode=args.mode,
               input_size=input_size,
               input_img_size=input_img_size,
               input_p_size=input_p_size,
               hidden_size=hidden_size,
               ctc=ctc_flg)
    else:
        from utils.trainer import trainer
        from models.model import Basemodel
        net = Basemodel(mode=args.mode,
               input_size=input_size,
               input_img_size=input_img_size,
               input_p_size=input_p_size,
               hidden_size=hidden_size,
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
