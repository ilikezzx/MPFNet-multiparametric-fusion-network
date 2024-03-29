import os
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()


def save_checkpoint(state, is_best, checkpoint_dir, logger=None):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
        If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.
        Args:
            state (dict): contains model's state_dict, optimizer's state_dict, epoch
                and best evaluation metric value so far
            is_best (bool): if True state contains the best model seen so far
            checkpoint_dir (string): directory where the checkpoint are to be saved
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


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Loads model and training parameters from a given checkpoint_path
       If optimizer is provided, loads optimizer's state_dict of as well.
       Args:
           checkpoint_path (string): path to the checkpoint to be loaded
           model (torch.nn.Module): model into which the parameters are to be copied
           optimizer (torch.optim.Optimizer) optional: optimizer instance into
               which the parameters are to be copied
       Returns:
           state
       """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer_state_dict'])

    return state
