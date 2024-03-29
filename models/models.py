"""
# File       : models.py
# Time       ：2022/8/7 14:42
# Author     ：zzx
# version    ：python 3.10
# Description：
    Basic Model Class
"""

import os
import shutil
import torch
import torch.nn as nn


class ModuleClass(nn.Module):
    def __init__(self):
        super(ModuleClass, self).__init__()

    def save_checkpoint(self, state, is_best, checkpoint_dir, logger=None):
        """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
            If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.
            Args:
                state (dict): contains model's state_dict, optimizer's state_dict, epoch
                    and best evaluation metric value so far
                is_best (bool): if True state contains the best model seen so far
                checkpoint_dir (string): directory where the checkpoint are to be saved
                logger: print logger
            """

        def log_info(message):
            if logger is not None:
                logger.info(message)

        if not os.path.exists(checkpoint_dir):
            log_info(f"Checkpoint directory does not exists. Creating {checkpoint_dir}")
            os.mkdir(checkpoint_dir)
        last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pt')
        log_info(f"Saving last checkpoint to '{last_file_path}'")
        torch.save(state, last_file_path)
        if is_best:
            best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pt')
            log_info(f"Saving best checkpoint to '{best_file_path}'")
            shutil.copyfile(last_file_path, best_file_path)

    def load_checkpoint(self, checkpoint_path, optimizer=None):
        """Loads model and training parameters from a given checkpoint_path
           If optimizer is provided, loads optimizer's state_dict of as well.
           Args:
               checkpoint_path (string): path to the checkpoint to be loaded
               optimizer (torch.optim.Optimizer) optional: optimizer instance into
                   which the parameters are to be copied
           Returns:
               state
           """
        if not os.path.exists(checkpoint_path):
            raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

        state = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(state['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(state['optimizer_state_dict'])

        return state
