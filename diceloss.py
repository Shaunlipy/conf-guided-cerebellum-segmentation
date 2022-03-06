import torch
from torch import nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, args):
        """:arg
        label starts with 1
        """
        super(DiceLoss, self).__init__()
        self.ignore_index = args.ignore_index
        self.smooth = 0.0
        self.eps = 1e-7
        self.device = args.device

    @staticmethod
    def soft_dice_score(output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        assert output.size() == target.size()
        if dims is not None:
            intersection = torch.sum(output * target, dim=dims)
            cardinality = torch.sum(output + target, dim=dims)
        else:
            intersection = torch.sum(output * target)
            cardinality = torch.sum(output + target)
        dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
        return dice_score


    def forward(self, pred, true):
        bs = true.size(0)
        num_classes = pred.size(1)
        dims = (0, 2)
        # pred = pred.log_softmax(dim=1).exp()
        pred = torch.exp(torch.nn.functional.log_softmax(pred, dim=1))
        true = true.view(bs, -1)
        pred = pred.view(bs, num_classes, -1)
        if self.ignore_index is not None:
            mask = true != self.ignore_index
            pred = pred * mask.unsqueeze(1)
            try:
                true = F.one_hot((true * mask).to(torch.long), num_classes)  # N,H*W -> N,H*W, C
            except:
                breakpoint()
            true = true.permute(0, 2, 1) * mask.unsqueeze(1)  # H, C, H*W
        else:
            true = F.one_hot(true, num_classes)  # N,H*W -> N,H*W, C
            true = true.permute(0, 2, 1)  # H, C, H*W
        scores = self.soft_dice_score(pred, true.type_as(pred), smooth=self.smooth, eps=self.eps, dims=dims)
        loss = 1.0 - scores
        mask = true.sum(dims) > 0
        loss *= mask.to(loss.dtype)
        return loss.mean()


if __name__ == '__main__':
    class Args:
        device = 'cpu'
        ignore_index = 0
    args = Args()
    pred = torch.randn(2, 5, 10, 10)
    gt = torch.ones(2, 10, 10).long()
    loss = DiceLoss(args)
    output = loss(pred, gt)
    print(output)