# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import random
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from prediction.model_builder import build_model
import recording.util as util


def generate_movie(num_frames, overlay=False):
    config.DATALOADER_TEST_SKIP_FRAMES = 1
    config.VAL_FILTER = ['data/test/*/*']   # By default the dataloader will return the validation set. Force to use the test set

    random.seed(5)  # Set the seed so the sequences will be randomized the same
    model_dict = build_model(config, device, ['val'])

    best_model = torch.load(util.find_latest_checkpoint(config))
    best_model.eval()

    val_dataloader = DataLoader(model_dict['val_dataset'], batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS)

    out_path = os.path.join('data', 'movies', config.CONFIG_NAME + '_movie.avi')
    print('Saving output video to:', out_path)
    util.mkdir(out_path, cut_filename=True)
    mw = util.MovieWriter(out_path)

    for idx, batch in enumerate(tqdm(val_dataloader, total=num_frames)):
        image_model = batch['img']
        image_original = batch['img_original']
        force_gt = batch['raw_force']
        participant = batch['participant'][0]
        action = batch['action'][0]

        with torch.no_grad():
            if config.FORCE_CLASSIFICATION:
                force_pred_class = best_model(image_model.cuda())
                force_pred_class = torch.argmax(force_pred_class, dim=1)
                force_pred_scalar = util.classes_to_scalar(force_pred_class, config.FORCE_THRESHOLDS)
            else:
                force_pred_scalar = best_model(image_model.cuda()).squeeze(1) * config.NORM_FORCE_REGRESS

            image_save = image_original.detach().squeeze().cpu().numpy().transpose((1, 2, 0))
            image_save = cv2.cvtColor(image_save * 255, cv2.COLOR_BGR2RGB).astype(np.uint8)

            force_pred_scalar[force_pred_scalar < 0] = 0    # Clip in case of negative values
            force_color_gt = util.pressure_to_colormap(force_gt.detach().squeeze().cpu().numpy())
            force_color_pred = util.pressure_to_colormap(force_pred_scalar.detach().squeeze().cpu().numpy())

            if overlay:
                val_img = 0.6
                force_color_gt = cv2.addWeighted(force_color_gt, 1.0, image_save, val_img, 0)
                force_color_pred = cv2.addWeighted(force_color_pred, 1.0, image_save, val_img, 0)

            disp_frame = np.zeros((384 * 2, 480 * 2, 3), np.uint8)
            util.set_subframe(0, image_save, disp_frame, title='RGB')
            util.set_subframe(1, force_color_gt, disp_frame, title='GT Pressure')
            util.set_subframe(2, image_save, disp_frame, title='RGB')
            util.set_subframe(3, force_color_pred, disp_frame, title='Estimated Pressure')

            cv2.putText(disp_frame, '{} {}'.format(participant, batch['timestep'][0].item()), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            cv2.putText(disp_frame, '{}'.format(action), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

            mw.write_frame(disp_frame)

        if idx >= num_frames:
            break

    mw.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=2000)
    parser.add_argument('--overlay', action='store_true')
    parser.add_argument('-cfg', '--config', type=str)
    args = parser.parse_args()
    config = util.load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generate_movie(args.frames, overlay=args.overlay)
