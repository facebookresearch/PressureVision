# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import argparse
import glob
import random
import os
import recording.util as util
import json
from functools import lru_cache


class SequenceReader:
    def __init__(self, seq_path):
        self.seq_path = seq_path

        split_path = os.path.normpath(seq_path).split(os.path.sep)
        self.action = split_path[-1]
        self.participant = split_path[-2]
        self.has_left, self.has_right = self.sequence_to_handedness(seq_path)

        self.metadata = util.json_read(os.path.join(self.seq_path, 'metadata.json'))
        self.num_cameras = self.metadata['num_cameras']
        self.num_frames = self.metadata['num_frames']
        self.timesteps = self.metadata['timesteps']
        self.lighting = self.get_lighting()
        self.skin_tone = self.metadata['participant']['Skintone']

        self.img_height = [1080, 1080, 1080, 1080]
        self.img_width = [1920, 1920, 1920, 1920]

        self.sensel_homography = [self.get_sensel_homography(c)[0] for c in range(self.num_cameras)]
        self.sensel_points = [self.get_sensel_homography(c)[1] for c in range(self.num_cameras)]

    def get_lighting(self):
        long_str = self.metadata['participant']['Light Direction']

        out_str = ''
        if 'left' in long_str:
            out_str += 'L'
        if 'center' in long_str:
            out_str += 'C'
        if 'right' in long_str:
            out_str += 'R'

        return out_str

    def sequence_to_handedness(self, seq):
        special_cases = {   # Left, right
            'type_sentence_5x_left': (True, True),
            'type_sentence_5x_right': (True, True),
            'type_sentence_5x_both': (True, True),
            'type_sentence_5x_both(2)': (True, True),
            'type_ipad_5x_both': (True, False),         # LEFT
            'type_ipad_5x_both(2)': (False, True),         # RIGHT
        }

        parts = os.path.normpath(seq).split(os.sep)     # Split to get the individual path parts
        seq_name = parts[-1]
        # seq_name = '_'.join(seq_name.split('_')[:-1])

        for s in special_cases:
            if seq_name == s:
                return special_cases[seq_name]

        if 'left' in seq_name:
            return True, False
        elif 'right' in seq_name:
            return False, True
        elif 'left' in seq:
            return True, False
        elif 'right' in seq:
            return False, True
        else:
            raise Exception('No handedness found')

    def get_img(self, camera_idx, frame_idx, to_rgb=False):
        img = cv2.imread(self.get_img_path(camera_idx, frame_idx))
        if to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def crop_img(self, img, camera_idx, config):
        min_xy = self.sensel_points[camera_idx].min(axis=0) - config.CROP_MARGIN    # These are XY
        max_xy = self.sensel_points[camera_idx].max(axis=0) + config.CROP_MARGIN
        center_xy = np.round((min_xy + max_xy) / 2).astype(int)

        # img_radius = np.round((max_xy - min_xy).max() / 2).astype(int)
        span_x = (max_xy[0] - min_xy[0]) / 2
        span_y = (max_xy[1] - min_xy[1]) / 2
        if span_x * config.NETWORK_IMAGE_SIZE_Y / config.NETWORK_IMAGE_SIZE_X > span_y:     # Y is smaller
            span_y = span_x * config.NETWORK_IMAGE_SIZE_Y / config.NETWORK_IMAGE_SIZE_X     # set Y to be based on X
        else:
            span_x = span_y * config.NETWORK_IMAGE_SIZE_X / config.NETWORK_IMAGE_SIZE_Y

        min_x = max(int(center_xy[0] - span_x), 0)
        max_x = min(int(center_xy[0] + span_x), img.shape[1])
        min_y = max(int(center_xy[1] - span_y), 0)
        max_y = min(int(center_xy[1] + span_y), img.shape[0])

        network_image_size = (config.NETWORK_IMAGE_SIZE_X, config.NETWORK_IMAGE_SIZE_Y)
        out_img = cv2.resize(img[min_y:max_y, min_x:max_x, ...], network_image_size)    # image is YX

        return out_img

    def get_force_pytorch(self, camera_idx, frame_idx, config):
        force = self.get_force_warped_to_img(camera_idx, frame_idx).astype('float32')

        if config.CROP_IMAGES is True:
            force = self.crop_img(force, camera_idx, config)
        return force

    def get_img_pytorch(self, camera_idx, frame_idx, config):
        """
        Helper function to get images in a pytorch-friendly format
        """
        img = self.get_img(camera_idx, frame_idx, to_rgb=True).astype('float32') / 255

        if config.CROP_IMAGES is True:
            img = self.crop_img(img, camera_idx, config)
        return img

    def get_depth(self, frame_idx):
        img_path = os.path.join(self.seq_path, 'depth', '{:05d}.png'.format(frame_idx))
        img = cv2.imread(img_path, -1)

        return img  # millimeters

    def get_img_path(self, camera_idx, frame_idx):
        return os.path.join(self.seq_path, 'camera_{}'.format(camera_idx), '{:05d}.jpg'.format(frame_idx))

    def get_pressure_kPa(self, frame_idx):
        pkl_path = os.path.join(self.seq_path, 'force', '{:05d}.pkl'.format(frame_idx))
        with open(pkl_path, 'rb') as handle:
            raw_counts = pickle.load(handle)

        kPa = util.convert_counts_to_kPa(raw_counts)
        return kPa

    def get_camera_params(self, camera_idx):
        params = dict()
        c = self.metadata['camera_calibrations'][str(camera_idx)]
        params['extrinsic'] = np.array(c['ModelViewMatrix'])
        params['intrinsic'] = np.array([[c['fx'], 0, c['cx']], [0, c['fy'], c['cy']], [0, 0, 1]])
        params['distortion'] = np.array([c['k1'], c['k2'], c['p1'], c['p2'], c['k3'], c['k4'], c['k5'], c['k6']])
        return params

    def get_sensel_homography(self, camera_idx):
        """
        Gets the 3x3 homography matrix to transform a point in sensel space into image space
        """

        camera_params = self.get_camera_params(camera_idx)

        rvec, jacobian = cv2.Rodrigues(camera_params['extrinsic'][:3, :3])
        tvec = camera_params['extrinsic'][:3, 3] / 1000.0  # Convert to meters

        sensel_w = 0.235
        sensel_h = 0.135
        sensel_origin_x = 0.016
        sensel_origin_y = 0.014
        sensel_z = -0.001

        sensel_corners_2D = np.float32([[185, 0], [185, 105], [0, 105], [0, 0]])
        sensel_corners_3D = np.float32([[sensel_origin_y + sensel_h, sensel_z, sensel_origin_x + sensel_w],
                                        [sensel_origin_y, sensel_z, sensel_origin_x + sensel_w],
                                        [sensel_origin_y, sensel_z, sensel_origin_x],
                                        [sensel_origin_y + sensel_h, sensel_z, sensel_origin_x]])

        points = np.array(sensel_corners_3D)
        image_points, jacobian = cv2.projectPoints(points, rvec, tvec, camera_params['intrinsic'], np.array([]))

        homography, status = cv2.findHomography(sensel_corners_2D, image_points[:, 0, :])
        return homography, image_points[:, 0, :]

    def get_force_warped_to_img(self, camera_idx, frame_idx, draw_sensel=False):
        force_img = self.get_pressure_kPa(frame_idx)
        force_warped = cv2.warpPerspective(force_img, self.sensel_homography[camera_idx], (self.img_width[camera_idx], self.img_height[camera_idx]))

        if draw_sensel:
            for c_idx in range(4):  # Draw the four corners on the image
                start_point = tuple(self.sensel_points[camera_idx][c_idx, :].astype(int))
                end_point = tuple(self.sensel_points[camera_idx][(c_idx + 1) % 4, :].astype(int))
                cv2.line(force_warped, start_point, end_point, 2, 5)

        return force_warped

    def get_force_overlay_img(self, camera_idx, frame_idx):
        force_warped = self.get_force_warped_to_img(camera_idx, frame_idx)
        force_color_warped = util.pressure_to_colormap(force_warped, colormap=cv2.COLORMAP_OCEAN)

        img = self.get_img(camera_idx, frame_idx)

        return cv2.addWeighted(img, 1.0, force_color_warped, 1.0, 0.0)

    def get_overall_frame(self, frame_idx, overlay_force=True):
        """
        Returns a frame with all views and cameras rendered as subwindows
        :return: A numpy array
        """
        out_x = 1440  # Rendering X, y
        out_y = 810
        subframes_x = 3
        subframes_y = 3

        disp_frame = np.zeros((out_y, out_x, 3), np.uint8)

        force = self.get_pressure_kPa(frame_idx)
        util.set_subframe(0, util.pressure_to_colormap(force), disp_frame, subframes_x, subframes_y, title='Pressure')

        depth = np.clip(self.get_depth(frame_idx) / 1500, 0, 1) * 255
        depth_color = cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_HOT)
        util.set_subframe(1, depth_color, disp_frame, subframes_x, subframes_y, title='Depth')

        for c in range(self.num_cameras):
            if overlay_force:
                img = self.get_force_overlay_img(c, frame_idx)
            else:
                img = self.get_img(c, frame_idx)
            util.set_subframe(c + 2, img, disp_frame, subframes_x, subframes_y, 'Camera {}'.format(c))

        return disp_frame
