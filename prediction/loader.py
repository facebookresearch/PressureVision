# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import pickle
import glob
from recording.sequence_reader import SequenceReader
from tqdm import tqdm
import torch
import recording.util as util
import random


def visualize_batch(image, pressure_gt, pressure_est=None):
    # Visualizes a batch
    image = util.to_cpu_numpy(image)[0, :].transpose(1, 2, 0)
    pressure_gt = util.to_cpu_numpy(pressure_gt)[0, :]

    plt.subplot(2, 2, 1)
    plt.gca().set_title('RGB Image')
    plt.imshow(image)

    plt.subplot(2, 2, 2)
    plt.gca().set_title('GT Pressure')
    plt.imshow(pressure_gt)

    if pressure_est is not None:
        pressure_est = util.to_cpu_numpy(pressure_est).squeeze()
        plt.subplot(2, 2, 3)
        plt.gca().set_title('Est Pressure')
        plt.imshow(pressure_est)

    plt.show()


class ForceDataset(Dataset):
    def __init__(
            self,
            config,
            seq_filter,
            preprocessing_fn=None,
            image_method=0,
            force_method=False,
            skip_frames=1,
            randomize_cam_seq=True,
            phase='val'
    ):
        self.config = config
        self.skip_frames = skip_frames
        self.randomize_cam_seq = randomize_cam_seq

        print('Loading dataset with filter: {}. Skipping {} frames'.format(seq_filter, skip_frames))

        self.all_datapoints = self.load_sequences(seq_filter)

        print('Done loading dataset. Total size:', len(self.all_datapoints))
        if len(self.all_datapoints) == 0:
            raise ValueError('Tried to load dataset, didnt find anything')

        self.phase = phase
        self.image_method = image_method
        self.force_method = force_method
        self.preprocessing_fn = preprocessing_fn

    def __getitem__(self, i):
        timestep = self.all_datapoints[i]['timestep']
        camera_idx = self.all_datapoints[i]['camera_idx']
        seq_reader = self.all_datapoints[i]['seq_reader']

        network_image_size = (self.config.NETWORK_IMAGE_SIZE_X, self.config.NETWORK_IMAGE_SIZE_Y)
        force_array = seq_reader.get_force_pytorch(camera_idx, timestep, self.config)
        image_0 = seq_reader.get_img_pytorch(camera_idx, timestep, self.config)

        if self.image_method == 0:  # Normal image
            image_out = image_0

        elif self.image_method == 4:    # Single image, but black and white
            image_out = cv2.cvtColor(image_0 * 255, cv2.COLOR_BGR2GRAY) / 255
            image_out = np.repeat(image_out[:, :, np.newaxis], 3, axis=2)

        elif self.image_method == 6:    # Resolution subsampled to quarter
            image_out = cv2.resize(image_0, (0, 0), fx=0.25, fy=0.25)
            image_out = cv2.resize(image_out, (0, 0), fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST)

        if self.preprocessing_fn is not None:
            image_out = self.preprocessing_fn(image_out)

        force_array = cv2.resize(force_array, network_image_size)
        raw_force_array = force_array

        if self.force_method == 0:  # Scalar
            force_array = np.clip(force_array / self.config.NORM_FORCE_REGRESS, 0, 1)
            force_array = np.expand_dims(force_array, axis=0)
        elif self.force_method == 2:    # Classes with custom thresholds
            force_array = util.scalar_to_classes(force_array, self.config.FORCE_THRESHOLDS)

        out_dict = dict()
        out_dict['img'] = image_out.transpose(2, 0, 1).astype('float32')
        out_dict['img_original'] = image_0.transpose(2, 0, 1).astype('float32')
        out_dict['force'] = force_array
        out_dict['seq_path'] = seq_reader.seq_path
        out_dict['camera_idx'] = camera_idx
        out_dict['timestep'] = timestep
        out_dict['participant'] = seq_reader.participant
        out_dict['action'] = seq_reader.action
        out_dict['raw_force'] = raw_force_array

        return out_dict

    def __len__(self):
        return len(self.all_datapoints)

    def load_sequences(self, seq_filter, use_cameras=None):
        """
        Takes a filter pointing to many dataset sequence and creates an list of "datapoints", each which is a training sample
        """
        if not isinstance(seq_filter, list):
            raise ValueError('Need a sequence filter list!')

        datapoints = []

        all_sequences = []
        for filter in seq_filter:
            all_sequences.extend(glob.glob(filter))

        for seq_path in all_sequences:
            if any([exclude in seq_path for exclude in self.config.EXCLUDE_ACTIONS]):
                continue

            seq_reader = SequenceReader(seq_path)
            for c in range(seq_reader.num_cameras):
                if use_cameras is not None and c not in use_cameras:
                    continue

                this_camera_points = []
                for t in range(seq_reader.num_frames):
                    datapoint = dict()
                    datapoint['seq_reader'] = seq_reader
                    datapoint['camera_idx'] = c
                    datapoint['timestep'] = t
                    this_camera_points.append(datapoint)
                datapoints.append(this_camera_points)

        if self.randomize_cam_seq:
            random.shuffle(datapoints)

        flattened = [item for sublist in datapoints for item in sublist]

        flattened = flattened[::self.skip_frames]     # Take every n'th sample of the dataset according to skip_frames

        return flattened


if __name__ == "__main__":
    config = util.parse_config_args()
    test_dataset = ForceDataset(config, config.VAL_FILTER, image_method=config.DATALOADER_IMAGE_METHOD,
                                force_method=config.DATALOADER_FORCE_METHOD,
                                skip_frames=config.DATALOADER_TRAIN_SKIP_FRAMES)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)

    video_data = []

    for idx, batch in enumerate(tqdm(test_dataloader)):
        image_model = batch['img']
        force_gt = batch['force']
        visualize_batch(image_model, force_gt)
