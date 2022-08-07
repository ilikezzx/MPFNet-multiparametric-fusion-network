import torch
import torch.nn.functional as F
from tqdm import tqdm

from metrics import Metrics, MultiMetrics
from utils.dice_score import multiclass_dice_coeff, dice_coeff

T1_area_name = ['肿瘤区域', '坏死区域']
T2_area_name = ['肿瘤区域', '水肿区域']


def evaluate(net, dataloader, device, logging):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    other_val_metrics_t1 = Metrics()
    other_val_metrics_t2 = Metrics()
    if net.n_classes > 1:
        other_val_metrics_t1 = MultiMetrics()
        other_val_metrics_t2 = MultiMetrics()

    channel_dsc_t1 = [0.0] * (net.n_classes - 1)
    channel_dsc_t2 = [0.0] * (net.n_classes - 1)

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        t1_image = batch['t1_image']
        t1_tg = batch['t1_mask']
        t2_image = batch['t2_image']
        t2_tg = batch['t2_mask']
        t1_image = t1_image.to(device=device, dtype=torch.float32)
        t1_tg = t1_tg.to(device=device, dtype=torch.long)
        t2_image = t2_image.to(device=device, dtype=torch.float32)
        t2_tg = t2_tg.to(device=device, dtype=torch.long)

        with torch.no_grad():
            # predict the mask
            t1_pred, t2_pred = net(t1_image, t2_image)

            mask_pred_oh_t1 = F.one_hot(t1_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            mask_true_oh_t1 = F.one_hot(t1_tg, net.n_classes).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background

            batch_dsc_t1, batch_channel_dsc_t1 = multiclass_dice_coeff(mask_pred_oh_t1[:, 1:, ...],
                                                                       mask_true_oh_t1[:, 1:, ...],
                                                                       reduce_batch_first=False, is_test=True)

            mask_pred_oh_t2 = F.one_hot(t2_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            mask_true_oh_t2 = F.one_hot(t2_tg, net.n_classes).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background

            batch_dsc_t2, batch_channel_dsc_t2 = multiclass_dice_coeff(mask_pred_oh_t2[:, 1:, ...],
                                                                       mask_true_oh_t2[:, 1:, ...],
                                                                       reduce_batch_first=False, is_test=True)

            dice_score += (batch_dsc_t1 + batch_dsc_t2) / 2

            for channel in range(len(batch_channel_dsc_t1)):
                channel_dsc_t1[channel] += batch_channel_dsc_t1[channel]

            for channel in range(len(batch_channel_dsc_t2)):
                channel_dsc_t2[channel] += batch_channel_dsc_t2[channel]

            other_val_metrics_t1.upd(t1_pred, t1_tg)
            other_val_metrics_t2.upd(t2_pred, t2_tg)

    for channel in range(len(channel_dsc_t1)):
        logging.info(f'T1: {T1_area_name[channel]} category dsc:{channel_dsc_t1[channel] / num_val_batches}')

    for channel in range(len(channel_dsc_t2)):
        logging.info(f'T2: {T2_area_name[channel]} category dsc:{channel_dsc_t2[channel] / num_val_batches}')

    net.train()
    return dice_score / num_val_batches, (other_val_metrics_t1.get(), other_val_metrics_t2.get())
