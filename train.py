import argparse
import os
import torch
from allendataset import AllenDataset
from bigdataset import BigDataset
from model import UnetPlus
from diceloss import DiceLoss

def parse_args():
    parser = argparse.ArgumentParser(description='Conf_guided_segmentation')
    parser.add_argument('--exp', type=int, required=True, choices=[0,1,2,3], help='indicate current iteration: 0 for AB training, 1 - 3 for iterative joint training')
    parser.add_argument('--conf_repeat', type=int, default=5, help='confidence guided variational sampling')
    # parser.add_argument('--training', type=str, default='', required=True,
                        # choices=['AllenDataset', 'Joint']) # 
    parser.add_argument('--dataroot', type=str, default='', help='file dir where all downloaded images are stored')
    parser.add_argument('--ratio', type=float, default=0.5,
                        help="percentage of A dataset in a batch when in dual training")
    parser.add_argument('--averagemeter', type=str, default='Meter_seg')
    parser.add_argument('--ignore_index', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=6) # num_classes + 1 ignored class / For allen brain 5
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--metric', type=str, default='accuracy')
    parser.add_argument('--crop_h', type=int, default=800)
    parser.add_argument('--crop_w', type=int, default=800)
    parser.add_argument('--img_h', type=int, default=512)
    parser.add_argument('--img_w', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--scheduler', type=str, default='Step')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--step_size', type=int, default=200)
    parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate decay multiplier')
    parser.add_argument('--lr_scheduler', type=str, choices=['StepLR'], default='StepLR')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda:0', 'cuda:1'], default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--out_dir', type=str, default='output')
    args = parser.parse_args()
    args.out_dir = os.path.join(args.out_dir, str(args.exp))
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
    if args.exp == 0:
        args.training = 'AllenDataset'
    else:
        args.training = 'Joint'
    return args


def get_data_loader(args, choice):
    if choice == 1:
        dset = AllenDataset(args)
        batch_size = int(args.batch_size * args.ratio)
    elif choice == 2:
        dset = BigDataset(args)
        batch_size = args.batch_size - int(args.batch_size * args.ratio)
    dloader = torch.utils.data.DataLoader(
        dset, batch_size=batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    return dloader


def train_Joint(epoch):
    model.train()
    dloader_1 = iter(train_loader_1)
    for i, data in enumerate(train_loader_2):
        input_2 = data['x'].to(args.device)  # batch, 3, 256, 256
        target_2 = data['y'].to(args.device)
        input_2 = input_2.view(-1, args.in_channels, args.img_h, args.img_w)
        target_2 = target_2.view(-1, args.img_h, args.img_w)
        output_2 = model(input_2, conf=True)
        loss_2 = criterion(output_2, target_2)
        try:
            data_1 = next(dloader_1)
        except StopIteration:
            dloader_1 = iter(train_loader_1)
            data_1 = next(dloader_1)
        input_1 = data_1['x'].to(args.device)  # batch, 3, 256, 256
        target_1 = data_1['y'].to(args.device)
        input_1 = input_1.view(-1, args.in_channels, args.img_h, args.img_w)
        target_1 = target_1.view(-1, args.img_h, args.img_w)
        output_1 = model(input_1)
        loss_1 = criterion(output_1, target_1)  # , target_cls)
        optimizer.zero_grad()
        loss = loss_1 + loss_2
        loss.backward()
        if i%50 == 0:
            print(f'Training {i} of epoch {epoch}|loss1 {loss_1.item():.4f}|loss2 {loss_2.item():.4f}')

def train_AllenDataset(epoch):
    model.train()
    for i, data in enumerate(train_loader_1):
        input_1 = data['x'].to(args.device)  # batch, 3, 256, 256
        target_1 = data['y'].to(args.device)
        input_1 = input_1.view(-1, args.in_channels, args.img_h, args.img_w)
        target_1 = target_1.view(-1, args.img_h, args.img_w)
        output_1 = model(input_1)
        loss = criterion(output_1, target_1)  # , target_cls)
        optimizer.zero_grad()
        loss.backward()
        if i%50 == 0:
            print(f'Training {i} of epoch {epoch}|loss {loss.item():.4f}')

if __name__ == '__main__':
    args = parse_args()
    _train = eval(f'train_{args.training}')
    train_loader_1 = get_data_loader(args, choice=1)
    train_loader_2 = get_data_loader(args, choice=2)
    model = UnetPlus(args=args)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
    criterion = DiceLoss(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    model.to(args.device)
    for epoch in range(args.epochs):
        _train(epoch)
        scheduler.step()
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model.pt'))

