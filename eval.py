import torch
import torch.nn.functional as F
from tqdm import tqdm
from dice_loss import dice_coeff


def eval_net(net, loader, device, n_val, class_weights=None):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['mask']

            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)

            mask_pred = net(imgs)

            for true_mask, pred in zip(true_masks, mask_pred):
                if net.n_classes > 1:
                    tot += F.cross_entropy(
                        input=pred.unsqueeze(dim=0),
                        target=true_mask,
                        weight=class_weights.to(device=device)
                        ).item()
                else:
                    pred = (pred > 0.5).float()
                    tot += dice_coeff(pred, true_mask.squeeze(dim=1)).item()
            pbar.update(imgs.shape[0])
    return tot / n_val
