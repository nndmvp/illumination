import torch
import models as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import os
import numpy as np
from data.VIDataset import FusionDataset as FD


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/ddpm_test.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            print("Creating train dataloader.")
            test_dataset = FD(split='val',
                               crop_size=dataset_opt['resolution'],
                               ir_path='./data/data/illumination',
                               vi_path='./data/data/illumination',
                               is_crop=True)
            print("the training dataset is length:{}".format(test_dataset.length))
            val_loader = DataLoader(
                dataset=test_dataset,
                batch_size=1,
                shuffle=False,
                pin_memory=True,
                drop_last=True,
            )
            val_loader.n_iter = len(val_loader)

    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = 0

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    result_path = '{}/{}'.format(opt['path']
                                 ['results'], current_step)
    os.makedirs(result_path, exist_ok=True)
    for _, (val_data, _) in enumerate(val_loader):
        current_step += 1
        diffusion.feed_data2(val_data)
        diffusion.test_concat(in_channels=6,
                              img_size_w=opt['datasets']['val']['image_size_w'],
                              img_size_h=opt['datasets']['val']['image_size_h'],
                              continous=False)

        visuals = diffusion.get_current_visuals()
        sam_img = Metrics.tensor2img(visuals['SAM'][:, 0:3, :, :])

        # generation
        Metrics.save_img(
            sam_img, '{}/sample_{}_{}.png'.format(result_path, current_step, current_step))

