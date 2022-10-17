#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bone-segmentation-1 
@Author  ：zzx
@Date    ：2022/10/13 19:37 
"""

import warnings
import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable  # torch 中 Variable 模块
from ptflops import get_model_complexity_info
from tqdm import tqdm
from utils.dataLoading_registration import MyDataset
from torch.utils.tensorboard import SummaryWriter
from utils.dice_score import dice_loss

from evaluate import evaluate
from models import MP_TBNet, MP_TBNet_ADD, AttU_Net, U_Net_3D, UNet

writer = SummaryWriter('./log')
warnings.filterwarnings("ignore")

goal_batch_size = 32


def train_net(net,
              device,
              epochs: int = 50,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              is_concat: bool = False,
              is_3D: bool = True,
              amp: bool = False):
    # 1. Create dataset
    dataset = MyDataset(args.dataset_path)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    accumulation_step = goal_batch_size // batch_size

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        is_concat:       {is_concat}
        Mixed Precision: {amp}
        accumulation_step: {accumulation_step}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()  # 多分类
    # criterion = torch.nn.BCEWithLogitsLoss()
    global_step = 0
    best_score = 0.0
    start_epoch = 0

    if args.load:
        # net.load_state_dict(torch.load(args.load, map_location=device))
        state = net.load_checkpoint(args.load, optimizer)
        start_epoch = state['epoch']
        best_score = state['best_acc']

        logging.info(f'Model loaded from {args.load}, best dsc is {best_score}')

    # 5. Begin training
    for epoch in range(start_epoch, epochs):
        net.train()
        epoch_loss = 0
        tot_loss = 0.0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for index, batch in enumerate(train_loader):
                # images = batch['image']
                # true_masks = batch['mask']
                T1Image = batch['T1Image']
                T2Image = batch['T2Image']
                tg = batch['Mask']

                assert T1Image.shape[1] == 1, \
                    f'Network has been defined with {1} input channels, ' \
                    f'but loaded images have {T1Image.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                assert T2Image.shape[1] == 1, \
                    f'Network has been defined with {1} input channels, ' \
                    f'but loaded images have {T2Image.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                T1Image = T1Image.to(device=device, dtype=torch.float32)
                T2Image = T2Image.to(device=device, dtype=torch.float32)
                tg = tg.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    if is_concat:
                        input_images = torch.cat([T1Image, T2Image], dim=1)
                        pred = net(input_images)
                    else:
                        pred = net(T1Image, T2Image)

                    loss = 0.5 * criterion(pred, tg) + 0.5 * dice_loss(F.softmax(pred, dim=1).float(),
                                                                       F.one_hot(tg, net.n_classes).permute(0,
                                                                                                               3,
                                                                                                               1,
                                                                                                               2).float(),
                                                                       multiclass=True)

                    tot_loss += loss.item()

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()

                if (index + 1) % accumulation_step == 0 or (index + 1) == len(train_loader):
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    lr_scheduler.step()  # 更新学习率

                pbar.update(T1Image.shape[0])
                global_step += 1
                pbar.set_postfix(**{'avg loss': tot_loss / (index + 1)})

            writer.add_scalar('train_loss', tot_loss / len(train_loader), epoch)
            torch.cuda.empty_cache()
            # val_score, (other_val_metrics_t1, other_val_metrics_t2) = evaluate(net, val_loader, device, logging, is_concat=is_concat)
            val_score = evaluate(net, val_loader, device, logging, is_concat=is_concat)

            # logging
            logging.info(f'Validation Dice score: {val_score}')
            # logging.info(
            #     f'Other: Recall-{other_val_metrics["Recall"]}, Precision-{other_val_metrics["Precision"]},'
            #     f'F1-Score-{other_val_metrics["F1-score"]}')

            writer.add_scalar('val_dsc', val_score, epoch)
            # writer.add_scalar('val_recall', other_val_metrics["Recall"], epoch)
            # writer.add_scalar('val_precision', other_val_metrics["Precision"], epoch)
            # #         writer.add_scalar('val_specificity', other_val_metrics["Specificity"], epoch)
            # #         writer.add_scalar('val_accuracy', other_val_metrics["Acc"], epoch)
            # writer.add_scalar('val_fs', other_val_metrics["F1-score"], epoch)
            # #         writer.add_scalar('val_jc', other_val_metrics["Jaccard Coefficient"], epoch)

            is_best = True if val_score > best_score else False
            if is_best:
                best_score = val_score

            stat = {
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_score,
                'device': str(device)
            }

        # save checkpoint
        Path(args.checkpoint_path).mkdir(parents=True, exist_ok=True)
        net.save_checkpoint(stat, is_best, args.checkpoint_path, logging)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--dataset-path', type=str, default=r'C:\Users\12828\Desktop\osteosarcoma'
                                                            r'\2dDataset_registration_havebad')
    parser.add_argument('--checkpoint-path', type=str, default=Path('./checkpoints/ResUNet/'))

    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=2, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('--is-concat', default=True, help='concating input-images')
    parser.add_argument('--is-3D', default=False, help='use 3D Dataset')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=25.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    input_channel = 1 + 1
    out_channel = 2

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(1 + 1, out_channel)

    # macs, params = get_model_complexity_info(net, (1, 224, 224)*2, as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    # print('{:<30} {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30} {:<8}'.format('Number of parameters: ', params))
    net.to(device=device)
    logging.info(f'Network:\n'
                 f'\t{input_channel} input channels\n'
                 f'\t{out_channel} output channels (classes)\n')

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  val_percent=args.val / 100,
                  is_concat=args.is_concat,
                  is_3D=args.is_3D,
                  amp=args.amp)
    except KeyboardInterrupt:
        # torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
