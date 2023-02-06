# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import segmentation_models_pytorch as smp
from recording.util import resnet_preprocessor

import ssl  # Hack to get around SSL certificate of the segmentation_models_pytorch being out of date
ssl._create_default_https_context = ssl._create_unverified_context

def build_model(config, device, phases):
    from prediction.loader import ForceDataset  # hacky shit

    if config.FORCE_CLASSIFICATION:
        criterion = torch.nn.CrossEntropyLoss()
        out_channels = config.NUM_FORCE_CLASSES
    else:
        if hasattr(config, 'USE_MAE') and config.USE_MAE:
            criterion = torch.nn.L1Loss()
        else:
            criterion = torch.nn.MSELoss()
        out_channels = 1

    print('Loss function:', criterion)

    if config.NETWORK_TYPE == 'smp':
        ENCODER = 'se_resnext50_32x4d'
        ENCODER_WEIGHTS = 'imagenet'

        # create segmentation model with pretrained encoder
        model = smp.FPN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=out_channels,
            activation=None,    # raw logits
            in_channels=config.NETWORK_INPUT_CHANNELS
        )

        # preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
        preprocessing_fn = resnet_preprocessor
    else:
        raise ValueError('Unknown model')

    model = model.to(device)

    out_dict = dict()
    out_dict['model'] = model
    out_dict['criterion'] = criterion

    if 'train' in phases:
        out_dict['train_dataset'] = ForceDataset(config, config.TRAIN_FILTER,
                                                 image_method=config.DATALOADER_IMAGE_METHOD,
                                                 force_method=config.DATALOADER_FORCE_METHOD,
                                                 skip_frames=1,
                                                 preprocessing_fn=preprocessing_fn,
                                                 phase='train')

    if 'val' in phases:
        out_dict['val_dataset'] = ForceDataset(config, config.VAL_FILTER,
                                               image_method=config.DATALOADER_IMAGE_METHOD,
                                               force_method=config.DATALOADER_FORCE_METHOD,
                                               skip_frames=config.DATALOADER_TEST_SKIP_FRAMES,
                                               preprocessing_fn=preprocessing_fn,
                                               phase='val')

    return out_dict

