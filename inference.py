import argparse
import os
from pathlib import Path
import cv2
import numpy as np
import tqdm
import torch
from torch.utils.data import DataLoader
from bigdataset import BigDataset
from model import UnetPlus


def parse_args():
    parser = argparse.ArgumentParser(description='Conf_guided_segmentation inference to generate pseudo label for BB')
    parser.add_argument('--num_classes', type=int, default=6) # num_classes + 1 ignored class / For allen brain 6
    parser.add_argument('--ignore_index', type=int, default=0)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--img_h', type=int, default=512)
    parser.add_argument('--img_w', type=int, default=512)
    parser.add_argument('--dataroot', type=str, default='')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda:0', 'cuda:1'], default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--ckpt_dir', type=str, default='', help='path to the .pt file (in the format of output/0/model.pt')
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--use_tqdm', action='store_false')
    parser.add_argument('--tqdm_minintervals', type=float, default=2.0)
    parser.add_argument('--conf_repeat', type=int, default=5)

    args = parser.parse_args()
    args.exp = int(Path(args.ckpt_dir).parent.stem)
    return args


def get_model(args):
    model = UnetPlus(args=args)
    ckp = torch.load(args.ckpt_dir, map_location='cpu')
    model.load_state_dict(ckp)
    model.to(args.device)
    model.eval()
    del ckp
    return model


def get_data(args):
    dset = BigDataset(args, inf=True)
    dloader = torch.utils.data.DataLoader(
        dset, batch_size=args.batch_size, num_workers=args.num_workers)
    return dloader


if __name__ == '__main__':
    args = parse_args()
    model = get_model(args)
    dloader = get_data(args)
    if args.use_tqdm:
        dloader = tqdm.tqdm(dloader)
    with torch.no_grad():
        for data in dloader:
            img = data['x']
            files = data['file']
            ws = data['width']
            hs = data['height']
            img = img.to(args.device)
            output = model(img)
            output = torch.softmax(output, dim=1)
            probs, preds = output.max(1)
            for i in range(preds.shape[0]):
                save_vis = str(Path(files[i]).parent).replace('B_input', f'B_input/{args.exp}_vis')
                save_anno = str(Path(files[i]).parent).replace('B_input', f'B_input/{args.exp}_anno')
                if not os.path.exists(save_vis):
                    os.makedirs(save_vis, exist_ok=True)
                if not os.path.exists(save_anno):
                    os.makedirs(save_anno, exist_ok=True)
                pred = preds[i].cpu().numpy()
                pred = cv2.resize(pred, (int(ws[i]), int(hs[i])), interpolation=cv2.INTER_NEAREST)
                out_vis = np.zeros((*pred.shape, 3)).astype(np.uint8)
                out_vis[pred == 1] = [255, 255, 255]
                out_vis[pred == 2] = [255, 0, 0]
                out_vis[pred == 3] = [0, 255, 0]
                out_vis[pred == 4] = [0, 0, 255]
                out_vis[pred == 5] = [122, 122, 122]
                cv2.imwrite(os.path.join(save_vis, Path(files[i]).name), out_vis)
                cv2.imwrite(os.path.join(save_anno, Path(files[i]).name), pred)