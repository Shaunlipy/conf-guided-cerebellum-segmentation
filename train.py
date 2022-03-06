import argparse
import os
import torch
from .allendataset import AllenDataset
from .bigdataset import BigDataset
from .model import UnetPlus
from .diceloss import DiceLoss

def parse_args():
    parser = argparse.ArgumentParser(description='Conf_guided_segmentation')
    parser.add_argument('--training', type=str, default='', required=True,
                        choices=['AllenDataset', 'BigDataset', 'Joint'])
    parser.add_argument('--train_file_a', type=str, default='data/allen_brain/train_430_cycle.txt')
    parser.add_argument('--val_file_a', type=str, default='data/allen_brain/val_430_cycle.txt')
    parser.add_argument('--file_prefix_a', type=str, default='', help='path to the allendataset input directory')
    parser.add_argument('--anno_prefix_a', type=str, default='', help='path to the allendataset annotation directory')
    parser.add_argument('--train_file_b', type=str, default='data/big_brain/brain.txt')
    parser.add_argument('--val_file_b', type=str, default='data/big_brain/brain.txt')
    parser.add_argument('--file_prefix_b', type=str, default='', help='path to the bigbrain input directory')
    parser.add_argument('--anno_prefix_b', type=str, default='', help='path to the bigbrain pseudo label directory')
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
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    return args


def get_data_loader(args, mode, choice):
    if choice == 1:
        dset = AllenDataset(args, args.train_file_a, args.val_file_a, args.file_prefix_a, args.anno_prefix_a, mode)
        batch_size = int(args.batch_size * args.ratio)
    elif choice == 2:
        dset = BigDataset(args, args.train_file_b, args.val_file_b, args.file_prefix_b, args.anno_prefix_b, mode)
        batch_size = args.batch_size - int(args.batch_size * args.ratio)
    shuffle = True if mode == 'train' else False
    dloader = torch.utils.data.DataLoader(
        dset, batch_size=batch_size, shuffle=shuffle,
        num_workers=args.num_workers, pin_memory=True)
    return dloader


def train(epoch):
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
            print(f'Training epoch {epoch}|loss1 {loss_1.item():.4f}|loss2 {loss_2.item():.4f}')

@torch.no_grad()
def val(epoch):
    model.eval()
    dloader_1 = iter(val_loader_2)
    for i, data in enumerate(val_loader_2):
        input_2 = data['x'].to(args.device)  # batch, 3, 256, 256
        target_2 = data['y'].to(args.device)
        input_2 = input_2.view(-1, args.in_channels, args.img_h, args.img_w)
        target_2 = target_2.view(-1, args.img_h, args.img_w)
        output_2 = model(input_2, conf=True)
        loss_2 = criterion(output_2, target_2)
        try:
            data_1 = next(dloader_1)
        except StopIteration:
            dloader_1 = iter(val_loader_1)
            data_1 = next(dloader_1)
        input_1 = data_1['x'].to(args.device)  # batch, 3, 256, 256
        target_1 = data_1['y'].to(args.device)
        input_1 = input_1.view(-1, args.in_channels, args.img_h, args.img_w)
        target_1 = target_1.view(-1, args.img_h, args.img_w)
        output_1 = model(input_1)
        loss_1 = criterion(output_1, target_1)
        if i % 50 == 0:
            print(f'Val epoch {epoch}|loss1 {loss_1.item():.4f}|loss2 {loss_2.item():.4f}')

if __name__ == '__main__':
    args = parse_args()
    train_loader_1 = get_data_loader(args, 'train', choice=1)
    val_loader_1 = get_data_loader(args, 'val', choice=1)
    train_loader_2 = get_data_loader(args, 'train', choice=2)
    val_loader_2 = get_data_loader(args, 'val', choice=2)
    model = ResNet34UnetPlus(args=args)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
    criterion = DiceLoss(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    model.to(args.device)
    for epoch in range(args.epochs):
        train(epoch)
        val(epoch)
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model.pt'))

